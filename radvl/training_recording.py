import os
import torch
import timeit
import time
import datetime
from radvl.data_loading import prepare_data
import numpy as np
from torch import optim
from radvl.settings_recording import parameter_parser
from radvl.model_components import RADVL, seed_torch

from radvl.regularization_utils import get_L2reg, Myloss, kl_loss
from radvl.recorder_metadata import OriginalRecorder, sync_device
from radvl.recorder_base import append_csv, METRICS
from radvl.data_dataset import Dataset
from radvl import hypergraph as ConstructHW
import torch.nn.functional as F
from radvl.metrics import Metric_fun
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

import networkx as nx
from torch_geometric.data import Data



warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def k_matrix(matrix, k):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)


def pyg_similarity_graph(data, k):
    # 1) 计算药物-药物相似性矩阵 (kNN 或者其它方式)
    drdr_matrix = k_matrix(data, k)  # (N_dr, N_dr)

    # 2) 转成 networkx 图 (无向图)
    drdr_nx = nx.from_numpy_array(drdr_matrix)

    # 3) 提取边索引 edge_index
    drdr_edges = torch.tensor(list(drdr_nx.edges), dtype=torch.long).t().contiguous()  # [2, E]

    # 4) 提取边权重 edge_attr
    drdr_edge_weight = torch.tensor(
        [drdr_matrix[i, j] for i, j in drdr_nx.edges],
        dtype=torch.float
    )

    # 5) 节点特征
    drdr_x = torch.tensor(data, dtype=torch.float)  # [N_dr, F_dr]

    # 6) 构建 PyG Data 对象
    drdr_graph = Data(x=drdr_x, edge_index=drdr_edges, edge_attr=drdr_edge_weight)

    return drdr_graph

def train_epoch(model, train_data, optim, opt, k, recorder, fold):
    model.train()
    sync_device()
    training_started = time.perf_counter()

    regression_crit = Myloss(alpha=opt.alpha)

    one_index = train_data[2][0].to(device).t().tolist()
    zero_index = train_data[2][1].to(device).t().tolist()

    # Drug
    dis_sim_integrate_tensor = train_data[0].to(device)
    # circRNA
    mi_sim_integrate_tensor = train_data[1].to(device)


    #  circRNA-Drug 关联矩阵
    A = train_data[6].to(device)
    AT=A.T
    AT = AT.to(device)

    # 构建Drug的超图
    concat_dis = np.hstack(
        [train_data[4].numpy().T, dis_sim_integrate_tensor.detach().cpu().numpy()])
    concat_dis_tensor = torch.FloatTensor(concat_dis)
    concat_dis_tensor = concat_dis_tensor.to(device)

    G_dis_Kn = ConstructHW.constructHW_knn(concat_dis_tensor.detach().cpu().numpy(), K_neigs=[k], is_probH=False)
    G_dis_Km = ConstructHW.constructHW_kmean(concat_dis_tensor.detach().cpu().numpy(), clusters=[9])
    G_dis_Kn = G_dis_Kn.to(device)
    G_dis_Km = G_dis_Km.to(device)

    # 构建Drug的相似图
    dis_sim_graph = pyg_similarity_graph(dis_sim_integrate_tensor.detach().cpu().numpy(), k)

    # 构建circRNA的超图
    concat_miRNA = np.hstack(
        [train_data[4].numpy(), mi_sim_integrate_tensor.detach().cpu().numpy()])
    concat_mi_tensor = torch.FloatTensor(concat_miRNA)
    concat_mi_tensor = concat_mi_tensor.to(device)   #（788,1162）

    G_mi_Kn = ConstructHW.constructHW_knn(concat_mi_tensor.detach().cpu().numpy(), K_neigs=[k], is_probH=False)
    G_mi_Km = ConstructHW.constructHW_kmean(concat_mi_tensor.detach().cpu().numpy(), clusters=[9])
    G_mi_Kn = G_mi_Kn.to(device)
    G_mi_Km = G_mi_Km.to(device)

    # 构建circRNA的相似图
    mi_sim_graph = pyg_similarity_graph(mi_sim_integrate_tensor.detach().cpu().numpy(), k)

    loss_kl = kl_loss(218, 271)
    recorder.graph_inputs(fold, A, concat_mi_tensor, concat_dis_tensor,
                          G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km,
                          mi_sim_graph, dis_sim_graph)
    current_epoch = 0

    for epoch in range(1, opt.epoch + 1):
        sync_device()
        epoch_started = time.perf_counter()

        score,reconstruction_en, _, _, cov_reg, FA_loss = model(concat_mi_tensor, concat_dis_tensor,
                                        G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km,
                                        AT, A, dis_sim_graph, mi_sim_graph)
        reconstruction_en=reconstruction_en.to(torch.float32).to(device)
        reconstruction_en = torch.sigmoid(reconstruction_en)
        BCE = torch.nn.functional.binary_cross_entropy(reconstruction_en, A, reduction='mean')

        # print("mean grad:", model.z_node_mean.requires_grad, model.z_edge_mean.requires_grad)
        # print("log_std grad:", model.z_node_log_std.requires_grad, model.z_edge_log_std.requires_grad)

        loss_k = loss_kl(model.z_node_log_std, model.z_node_mean, model.z_edge_log_std, model.z_edge_mean)

        # loss_beta = BCE + 0.5 * loss_k
        loss_beta = BCE + 0.5 * loss_k
        recover_loss = regression_crit(one_index, zero_index, train_data[4].to(device), score)#double
        reg_loss = get_L2reg(model.parameters())#float


        # 协同正则化的损失
        lambda_cov_use = model.co_lambda

        co_loss = lambda_cov_use * cov_reg
        # print(111111111111111111)
        # print(co_loss)
        tol_loss = recover_loss  + 0.0001 * reg_loss + loss_beta + co_loss + FA_loss * 2
        # tol_loss = 0.0001 * reg_loss + loss_beta + co_loss + FA_loss
        # losses = [recover_loss, reg_loss, loss_beta]



        # print(losses)
        optim.zero_grad()
        tol_loss.backward()
        optim.step()
        sync_device()
        compute_seconds = time.perf_counter() - epoch_started
        recorder.epoch(fold, epoch, tol_loss, dict(
            prediction_mse=recover_loss, reconstruction_bce=BCE, kl_original=loss_k,
            weighted_kl=0.5*loss_k, explicit_l2_raw=reg_loss,
            weighted_explicit_l2=0.0001*reg_loss, covariance_raw=cov_reg,
            weighted_covariance=co_loss, alignment_raw=FA_loss,
            weighted_alignment=2*FA_loss), model, optim, compute_seconds)


    sync_device()
    training_seconds = time.perf_counter() - training_started
    inference_started = time.perf_counter()
    true_value_one, true_value_zero, pre_value_one, pre_value_zero = test(model, train_data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km, AT,A, dis_sim_graph, mi_sim_graph)
    sync_device()
    recorder.performance(fold, model, training_seconds, time.perf_counter()-inference_started)

    return true_value_one, true_value_zero, pre_value_one, pre_value_zero


def test(model, data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km,AT,A, dis_sim_graph, mi_sim_graph):
    model.eval()
    score, _, drug_emb, circ_emb, cov_reg_node, FA_loss= model(concat_mi_tensor, concat_dis_tensor,
                        G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km, AT, A, dis_sim_graph, mi_sim_graph)
    test_one_index = data[3][0].t().tolist()
    test_zero_index = data[3][1].t().tolist()
    true_one = data[5][test_one_index]
    true_zero = data[5][test_zero_index]

    pre_one = score[test_one_index]
    pre_zero = score[test_zero_index]
    # print(1111111111111111111111111111)
    # print(test_one_index)
    # print(test_zero_index)


    # print(test_pairs_one.shape) # [743, 128]
    # print(test_pairs_zero.shape) # [9889, 128]

    return true_one, true_zero, pre_one, pre_zero


def evaluate(true_one, true_zero, pre_one, pre_zero, save_dir, fold_i):
    Metric = Metric_fun()
    metrics_tensor = np.zeros((1, 7))

    all_fpr = []
    all_tpr = []
    all_auc = []

    all_precision = []
    all_recall = []
    all_aupr = []

    # test_po_num = true_one.shape[0]
    # test_index = np.array(np.where(true_zero == 0))
    # np.random.seed(1)
    # np.random.shuffle(test_index.T)
    # test_ne_index = tuple(test_index[:, :test_po_num])
    #
    # eval_true_zero = true_zero[test_ne_index]
    # eval_true_data = torch.cat([true_one, eval_true_zero])
    #
    # eval_pre_zero = pre_zero[test_ne_index]
    # eval_pre_data = torch.cat([pre_one, eval_pre_zero])

    # 正负样本数量在预处理阶段已保证一致
    eval_true_data = torch.cat([true_one, true_zero])
    eval_pre_data = torch.cat([pre_one, pre_zero])



    # 保存预测结果
    np.save(f'{save_dir}/true_{fold_i}.npy', eval_true_data.detach().cpu().numpy())
    np.save(f'{save_dir}/pred_{fold_i}.npy', eval_pre_data.detach().cpu().numpy())


    for seed in range(10):
        test_po_num = true_one.shape[0]
        test_index = np.array(np.where(true_zero == 0))
        np.random.seed(seed)
        np.random.shuffle(test_index.T)
        test_ne_index = tuple(test_index[:, :test_po_num])

        eval_true_zero = true_zero[test_ne_index]
        eval_true_data = torch.cat([true_one, eval_true_zero])

        eval_pre_zero = pre_zero[test_ne_index]
        eval_pre_data = torch.cat([pre_one, eval_pre_zero])

        seed_metrics = Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data)
        metrics_tensor = metrics_tensor + seed_metrics
        append_csv(os.path.join(save_dir, 'evaluation_seeds.csv'), [dict(
            fold=fold_i, evaluation_seed=seed, **dict(zip(METRICS, map(float, seed_metrics))))])

        # 计算 AUC、FPR 和 TPR
        fpr, tpr, _ = roc_curve(
            np.concatenate([np.ones(len(true_one)), np.zeros(len(eval_true_zero))]),
            np.concatenate([pre_one.detach().cpu().numpy(), eval_pre_zero.detach().cpu().numpy()])
        )
        roc_auc = auc(fpr, tpr)

        # 计算 PR 曲线
        precision, recall, _ = precision_recall_curve(
            np.concatenate([np.ones(len(true_one)), np.zeros(len(eval_true_zero))]),
            np.concatenate([pre_one.detach().cpu().numpy(), eval_pre_zero.detach().cpu().numpy()])
        )
        # 计算 AUC-PR
        pr_auc = auc(recall, precision)

        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_auc.append(roc_auc)

        all_precision.append(precision)
        all_recall.append(recall)
        all_aupr.append(pr_auc)

        # 输出每次评估的指标
        print(f'Seed {seed}, Metrics: {metrics_tensor}')

    metrics_tensor_avg = metrics_tensor / 10

    # 计算均值 FPR 和 TPR
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)], axis=0)
    mean_auc = np.mean(all_auc, axis=0)

    # 生成均匀的 Recall 点
    mean_recall = np.linspace(0, 1, 100)

    # 计算均值精确度
    mean_precision = np.mean(
        [np.interp(1 - mean_recall, 1 - recall, precision) for recall, precision in zip(all_recall, all_precision)],
        axis=0
    )

    # 计算均值 AUPR
    mean_aupr = np.mean(all_aupr)

    return metrics_tensor_avg, mean_fpr, mean_tpr, mean_auc, mean_precision, mean_recall, mean_aupr




def main(opt, k):
    dataset = prepare_data(opt)
    train_data = Dataset(opt, dataset)

    metrics_cross = np.zeros((1, 7))

    # 用于存储每一折的 ROC 数据
    all_fpr_per_fold = []
    all_tpr_per_fold = []
    all_auc_per_fold = []

    # 用于存储每一折的 PR 数据
    all_precision_per_fold = []
    all_recall_per_fold = []
    all_aupr_per_fold = []

    # Get current timestamp
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


    save_dir = r'./Results_original_record/' + 'cv_'+ str(opt.validation) + '/' + time_stamp + '/'
    recorder = OriginalRecorder(save_dir, opt, k, dataset)
    print('Results directory:', os.path.abspath(save_dir), flush=True)

    for i in range(opt.validation):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # 预测结果保存路径

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        hidden_list = [256, 256]
        num_proj_hidden = 64

        model = RADVL(args.mi_num, args.dis_num, hidden_list, num_proj_hidden, args, num_in_node=218, num_in_edge=271, num_hidden1=512, num_out=64, num_layers=4)
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=5e-4)

        # 多任务优化器
        # optimizer = IMTL(optimizer)

        true_score_one, true_score_zero, pre_score_one, pre_score_zero = train_epoch(model, train_data[i], optimizer, opt, k, recorder, i)
        metrics_value, mean_fpr,mean_tpr, mean_auc,mean_precision, mean_recall,mean_aupr = evaluate(true_score_one, true_score_zero, pre_score_one, pre_score_zero, save_dir, i)
        metrics_cross = metrics_cross + metrics_value
        recorder.save_fold(i, model, optimizer, opt, train_data[i], metrics_value)

        # 收集 ROC 数据
        all_fpr_per_fold.append(mean_fpr)
        all_tpr_per_fold.append(mean_tpr)
        all_auc_per_fold.append(mean_auc)

        # 收集 PR 数据
        all_precision_per_fold.append(mean_precision)
        all_recall_per_fold.append(mean_recall)
        all_aupr_per_fold.append(mean_aupr)


    metrics_cross_avg = metrics_cross / opt.validation
    print('metrics_avg:', metrics_cross_avg)
    recorder.finish()
    print('Results saved:', os.path.abspath(save_dir), flush=True)

    # # 绘制 ROC 曲线
    # plot_roc_curves(all_fpr_per_fold, all_tpr_per_fold, all_auc_per_fold)
    #
    #
    # #绘制 PR 曲线
    # plot_pr_curves(all_precision_per_fold, all_recall_per_fold, all_aupr_per_fold)



def plot_roc_curves(all_fpr_per_fold, all_tpr_per_fold, all_auc_per_fold):
    plt.figure(figsize=(10, 6))  # 设置图形大小

    plt.xlim([-0.05, 1.05])  # 将 x 轴的范围向右移动
    plt.ylim([-0.15, 1.15])  # 将 y 轴的下限设置为 -0.1，向下移动曲线

    mean_fpr = np.linspace(0, 1, 100)  # 均匀生成 100 个 FPR 点
    mean_tpr = np.zeros_like(mean_fpr)  # 创建数组用于存储均值 TPR

    # 绘制每一折的曲线
    for i in range(len(all_fpr_per_fold)):
        # 插值计算均值 TPR
        tpr_interpolated = np.interp(mean_fpr, all_fpr_per_fold[i], all_tpr_per_fold[i])
        tpr_interpolated[0] = 0  # 确保第一个点为0
        mean_tpr += tpr_interpolated  # 累加 TPR

        # 绘制每一折的 ROC 曲线
        plt.plot(all_fpr_per_fold[i], all_tpr_per_fold[i], linestyle='-',
                 label=f'Fold {i + 1}, Mean AUC = {all_auc_per_fold[i]:.6f}')

    mean_tpr /= len(all_fpr_per_fold)  # 计算均值 TPR
    mean_auc = np.mean(all_auc_per_fold)  # 计算均值 AUC

    plt.plot(mean_fpr, mean_tpr, linestyle='--', color='black',
             label=f'Mean ROC, AUC = {mean_auc:.6f}')

    plt.plot([0, 1], [0, 1], linestyle='--', color='red')  # 对角线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - 5-Fold Cross Validation')
    plt.legend(loc='lower right')

    # 去掉背景网格
    plt.grid(False)

    # 改成保存为 PDF
    plt.savefig('roc_curve5.pdf', dpi=300, bbox_inches='tight')
    plt.show()  # 显示图像


def plot_pr_curves(all_precision_per_fold, all_recall_per_fold, all_aupr_per_fold):
    plt.figure(figsize=(10, 6))  # 设置图形大小

    mean_recall = np.linspace(0, 1, 100)  # 均匀生成 100 个 Recall 点
    precision = np.zeros_like(mean_recall)  # 使用列表来存储均值 Precision

    # 绘制每一折的曲线
    for i in range(len(all_recall_per_fold)):
        # 插值计算均值 Precision
        precision_interpolated = np.interp(mean_recall, all_recall_per_fold[i], all_precision_per_fold[i])
        precision_interpolated[0] = 1.0  # 确保第一个点为1
        precision += precision_interpolated

        # 绘制每一折的 PR 曲线
        plt.plot(all_recall_per_fold[i], all_precision_per_fold[i], linestyle='-',
                 label=f'Fold {i + 1}, AUPR = {all_aupr_per_fold[i]:.6f}')

    precision /= len(all_precision_per_fold)    # 计算均值 Precision
    precision[-1] = 0  # 确保最后一个点为0
    mean_auc = np.mean(all_aupr_per_fold)  # 计算均值 AUC
    plt.plot(mean_recall, precision, linestyle='--', color='black',
             label=f'Mean PR, AUPR = {mean_auc:.6f}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - 5-Fold Cross Validation')
    plt.legend(loc='lower left')

    plt.grid(False)  # 去掉背景网格

    # 改成保存为 PDF
    plt.savefig('pr_curve5.pdf', dpi=300, bbox_inches='tight')
    plt.show()  # 显示图像




if __name__ == '__main__':
    args = parameter_parser()
    seed_torch(args.seed)
    # 邻居数量
    # for index, k in enumerate([1, 4, 8, 12, 16, 20, 24, 28, 32]):


    # 邻居数量k=3是最好的

    # HGCN层数的分析
    # for index, layer in enumerate([1, 2, 3, 4, 5]):
    # args.nlayer =  layer
    k = 8
    main(args, k)


    # # KAN-GCN层数的分析
    # for index, layer

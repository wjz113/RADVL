import csv
import os
import torch as t
import numpy as np
from math import e
import pandas as pd
from scipy import io
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return t.FloatTensor(md_data)


def read_mat(path, name):
    matrix = io.loadmat(path)
    matrix = t.FloatTensor(matrix[name])
    return matrix


def read_md_data(path, validation):
    result = [{} for _ in range(validation)]
    for filename in os.listdir(path):
        data_type = filename[filename.index('_')+1:filename.index('.')-1]
        num = int(filename[filename.index('.')-1])
        result[num-1][data_type] = read_csv(os.path.join(path, filename))
    return result


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)


def Gauss_M(adj_matrix, N):
    GM = np.zeros((N, N))
    rm = N * 1. / sum(sum(adj_matrix * adj_matrix))
    for i in range(N):
        for j in range(N):
            GM[i][j] = e ** (-rm * (np.dot(adj_matrix[i, :] - adj_matrix[j, :], adj_matrix[i, :] - adj_matrix[j, :])))
    return GM


def Gauss_D(adj_matrix, M):
    GD = np.zeros((M, M))
    T = adj_matrix.transpose()
    rd = M * 1. / sum(sum(T * T))
    for i in range(M):
        for j in range(M):
            GD[i][j] = e ** (-rd * (np.dot(T[i] - T[j], T[i] - T[j])))
    return GD


def prepare_data(opt):
    dataset = {}

    # dd_data = np.loadtxt(opt.data_path + '/disease_semantic_sim.txt')
    # dd_mat = np.array(dd_data)
    #
    # mm_data = np.loadtxt(opt.data_path + '/miRNA_functional_sim.txt')
    # mm_mat = np.array(mm_data)
    #
    # mi_dis_data = pd.read_csv(opt.data_path + '/mi_dis_mat_3.0.csv',index_col=0)

    dd_data = pd.read_csv(opt.data_path + '/drug_combined_sim.csv', index_col=0)
    # dd_data1 = pd.read_csv(opt.data_path + '/drug_str_sim.csv', index_col=0)
    dd_data1 = pd.read_csv(opt.data_path + '/integration_drug.csv', header=None)
    dd_data = (dd_data.values + dd_data1.values)/2.0
    dd_mat = np.array(dd_data)

    # mm_data = pd.read_csv(opt.data_path + '/mi_fun_sim_2.0.csv',index_col=0)
    mm_data = pd.read_csv(opt.data_path + '/gene_combined_sim.csv', index_col=0)
    # mm_data1 = pd.read_csv(opt.data_path + '/gene_seq_sim.csv', index_col=0)
    mm_data1 = pd.read_csv(opt.data_path + '/integration_circRNA.csv', header=None)

    mm_data = (mm_data.values + mm_data1.values)/2.0
    mm_mat = np.array(mm_data)
    #
    # mi_dis_data = pd.read_csv(opt.data_path + '/mi_dis_mat_2.0.csv',index_col=0)
    mi_dis_data = pd.read_csv(opt.data_path + '/association.csv', index_col=0)

    # mi_dis_data = np.loadtxt("data/association.txt")
    # mm_mat = np.loadtxt("data/integration_circRNA.txt")
    # dd_mat = np.loadtxt("data/integration_drug.txt")

    dataset['md_p'] = t.FloatTensor(np.array(mi_dis_data))
    dataset['md_true'] = dataset['md_p']
    dataset['md_v'] = dataset['md_p']


    all_zero_index = []
    all_one_index = []
    for i in range(dataset['md_p'].size(0)):
        for j in range(dataset['md_p'].size(1)):
            if dataset['md_p'][i][j] < 1:
                all_zero_index.append([i, j])
            if dataset['md_p'][i][j] >= 1:
                all_one_index.append([i, j])


    # np.random.seed(3042)
    np.random.seed(114514)
    np.random.shuffle(all_zero_index)
    np.random.shuffle(all_one_index)
    # print(11111111111111111)
    # print(len(all_one_index)) # 4134
    # print(len(all_zero_index)) # 54944

    # 保证负样本数量和正样本数量一致
    num_neg = len(all_one_index)  # 正样本数

    # num_neg = len(all_zero_index)  # 负样本数
    # num_neg = min(num_neg, num_pos)  # 取正负样本中的最小值

    # 随机负样本筛选
    all_zero_index = all_zero_index[:num_neg]

    # all_one_index = all_one_index[:num_neg]

    # print(len(all_one_index)) # 4134
    # print(len(all_zero_index)) # 54944

    # zero_tensor = t.LongTensor(all_zero_index)
    # zero_index = zero_tensor.split(int(zero_tensor.size(0) / 10), dim=0)
    #
    # one_tensor = t.LongTensor(all_one_index)
    # one_index = one_tensor.split(int(one_tensor.size(0) / 10), dim=0)
    #
    #
    # cross_zero_index = t.cat([zero_index[i] for i in range(9)])
    # cross_one_index = t.cat([one_index[j] for j in range(9)])
    #
    # new_zero_index = cross_zero_index.split(int(cross_zero_index.size(0) / opt.validation), dim=0)
    # new_one_index = cross_one_index.split(int(cross_one_index.size(0) / opt.validation), dim=0)

    # 转成tensor
    one_tensor = t.LongTensor(all_one_index)
    zero_tensor = t.LongTensor(all_zero_index)

    # 五折划分
    new_one_index = one_tensor.split(int(len(one_tensor) / 5), dim=0)
    new_zero_index = zero_tensor.split(int(len(zero_tensor) / 5), dim=0)

    dataset['md'] = []
    for i in range(opt.validation):
        a = [i for i in range(opt.validation)]
        if opt.validation != 1:
            del a[i]
        dataset['md'].append({'test': [new_one_index[i], new_zero_index[i]],
                              'train': [t.cat([new_one_index[j] for j in a]), t.cat([new_zero_index[j] for j in a])]})



    DGSM = Gauss_D(dataset['md_p'].numpy(), dataset['md_p'].size(1))
    MGSM = Gauss_M(dataset['md_p'].numpy(), dataset['md_p'].size(0))


    nd = mi_dis_data.shape[1]
    nm = mi_dis_data.shape[0]

    ID = np.zeros([nd, nd])

    for h1 in range(nd):
        for h2 in range(nd):
            if dd_mat[h1, h2] == 0:
                ID[h1, h2] = DGSM[h1, h2]
            else:
                ID[h1, h2] = (dd_mat[h1, h2] + DGSM[h1, h2]) / 2

    IM = np.zeros([nm, nm])

    for q1 in range(nm):
        for q2 in range(nm):
            if mm_mat[q1, q2] == 0:
                IM[q1, q2] = MGSM[q1, q2]
            else:
                IM[q1, q2] = (mm_mat[q1, q2] + MGSM[q1, q2]) / 2

    # dataset['ID'] = t.from_numpy(ID)
    # dataset['IM'] = t.from_numpy(IM)
    dataset['ID'] = t.from_numpy(dd_mat)
    dataset['IM'] = t.from_numpy(mm_mat)
    return dataset

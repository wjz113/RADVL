"""RADVL training protocol."""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from radvl.settings_protocol import parameter_parser
from radvl.data_protocol import prepare_protocol
from radvl.feature_inputs import load_external_inputs, compute_features, label_invariance_audit, build_graphs, forward
from radvl.protocol_adapter import ranking_metrics
from radvl.recorder_protocol import ProtocolRecorder
from radvl.recorder_metadata import sync_device
from radvl.recorder_base import append_csv, METRICS
from radvl.model_components import RADVL, seed_torch
from radvl.regularization_utils import Myloss, get_L2reg, kl_loss


def initial_state_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fit(model,optimizer,training,graphs,args,recorder):
    """No full A_true, test labels, or validation labels enter this interface."""
    model.train()
    device=graphs['A'].device
    positive=torch.tensor(training.positive_pairs.copy(),dtype=torch.long,device=device)
    negative=torch.tensor(training.negative_pairs.copy(),dtype=torch.long,device=device)
    pos=(positive[:,0],positive[:,1]); neg=(negative[:,0],negative[:,1])
    mask=torch.tensor(training.train_mask.copy(),dtype=torch.bool,device=device)
    mse=Myloss(alpha=args.alpha)
    kl=kl_loss(218,271)
    sync_device(); started=time.perf_counter()
    for epoch in range(1,args.epoch+1):
        sync_device(); epoch_started=time.perf_counter()
        score,reconstruction,_,_,covariance,alignment=forward(model,graphs)
        reconstruction=torch.sigmoid(reconstruction.float())
        bce=torch.nn.functional.binary_cross_entropy(reconstruction[mask],graphs['A'][mask],reduction='mean')
        loss_kl=kl(model.z_node_log_std,model.z_node_mean,model.z_edge_log_std,model.z_edge_mean)
        loss_mse=mse(pos,neg,graphs['A'],score)
        l2=get_L2reg(model.parameters())
        co_loss=model.co_lambda*covariance
        total=loss_mse+0.0001*l2+bce+0.5*loss_kl+co_loss+2*alignment
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        sync_device()
        recorder.epoch(training.fold,epoch,total,dict(prediction_mse=loss_mse,
            reconstruction_bce=bce,kl_original=loss_kl,weighted_kl=0.5*loss_kl,
            explicit_l2_raw=l2,weighted_explicit_l2=0.0001*l2,covariance_raw=covariance,
            weighted_covariance=co_loss,alignment_raw=alignment,weighted_alignment=2*alignment),
            model,optimizer,time.perf_counter()-epoch_started)
    sync_device()
    return time.perf_counter()-started


@torch.no_grad()
def predict(model,graphs):
    model.eval()
    return forward(model,graphs)[0]


def main(args):
    table,circ_static,drug_static,candidates=load_external_inputs(args)
    protocol=prepare_protocol(args.protocol_dir,Path(args.data_path)/'association.csv',candidates,
        folds=args.validation,split_seed=args.split_seed,scenario=args.scenario,
        train_fraction=args.train_fraction,train_negative_ratio=args.train_negative_ratio,
        test_negative_ratio=args.test_negative_ratio,validation_fraction=args.validation_fraction)
    print('Frozen protocol:',protocol.directory,'fits:',len(protocol),flush=True)
    if args.prepare_only:
        print('Protocol created/verified. No training requested.',flush=True)
        return
    recorded_args=copy.copy(args)
    recorded_args.validation=len(protocol)
    dataset=dict(md=[],md_true=torch.tensor(table.values,dtype=torch.float32),association=table,
                 candidates=protocol.candidates,folds=[None]*len(protocol))
    for fold in range(len(protocol)):
        values=protocol.arrays(fold)
        dataset['md'].append(dict(train=[torch.from_numpy(values['train_positive']),torch.from_numpy(values['train_negative'])],
            validation=[torch.from_numpy(values['validation_positive']),torch.from_numpy(values['validation_negative'])],
            test=[torch.from_numpy(values['test_positive']),torch.from_numpy(values['test_negative'])]))
    timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory=Path('Results_shared_protocol')/args.scenario/timestamp
    recorder=ProtocolRecorder(directory,recorded_args,8,dataset,protocol,args)
    print('Results:',directory.resolve(),flush=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold in range(len(protocol)):
        training=protocol.training(fold)
        features=compute_features(training,circ_static,drug_static)
        invariant=label_invariance_audit(training,circ_static,drug_static,
                                       protocol.evaluation(fold,'test'),features)
        recorder.write_json(f'label_invariance_{fold}.json',invariant)
        recorder.fold_inputs(fold,features)
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        graphs=build_graphs(features,8,device)
        recorder.graph_inputs(fold,graphs['A'],graphs['circ_features'],graphs['drug_features'],
            graphs['circ_knn'],graphs['circ_kmeans'],graphs['drug_knn'],graphs['drug_kmeans'],
            graphs['circ_graph'],graphs['drug_graph'])
        fold_seed=args.seed+fold
        seed_torch(fold_seed)
        model=RADVL(args.mi_num,args.dis_num,[256,256],64,args,
                     num_in_node=218,num_in_edge=271,num_hidden1=512,num_out=64,num_layers=4).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001,weight_decay=5e-4)
        if optimizer.state: raise AssertionError('Optimizer was not freshly initialized')
        append_csv(directory/'fold_initialization.csv',[dict(fold=fold,training_seed=fold_seed,
            initial_model_sha256=initial_state_hash(model),optimizer_state_entries=0,
            training_fingerprint=training.fingerprint,cache_policy='fresh graphs, model, optimizer')])
        train_seconds=fit(model,optimizer,training,graphs,args,recorder)
        sync_device(); inference_started=time.perf_counter()
        scores=predict(model,graphs)
        sync_device(); inference_seconds=time.perf_counter()-inference_started
        # Test labels are consumed only here, after optimization has finished.
        test=protocol.evaluation(fold,'test')
        scores_numpy=scores.detach().cpu().numpy()
        predictions=scores_numpy[test.pairs[:,0],test.pairs[:,1]]
        values=ranking_metrics(test.labels,predictions,threshold=0.5)
        metrics=np.array([[values[name] for name in METRICS]])
        np.save(directory/f'true_{fold}.npy',test.labels)
        np.save(directory/f'pred_{fold}.npy',predictions)
        split=dataset['md'][fold]
        recording_view=(features['ID'],features['IM'],split['train'],split['test'],
                        features['A'],dataset['md_true'],features['A'],features['reconstruction_mask'])
        recorder.performance(fold,model,train_seconds,inference_seconds)
        recorder.save_fold(fold,model,optimizer,recorded_args,recording_view,metrics)
        print('Fold',fold,'seed',fold_seed,values,flush=True)
        del scores,model,optimizer,graphs,features
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    recorder.finish()
    print('Results saved:',directory.resolve(),flush=True)


if __name__=='__main__':
    main(parameter_parser())

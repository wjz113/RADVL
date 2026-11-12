"""RADVL feature inputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from radvl.data_preprocessing import gip, entropy_similarity, read_static
from radvl.data_protocol import array_hash


def load_external_inputs(args):
    root=Path(args.data_path)
    table=pd.read_csv(root/'association.csv',index_col=0)
    if table.shape!=(args.dis_num,args.mi_num):
        raise ValueError('The original DGAA model requires its configured entity dimensions')
    circ=read_static(root/'gene_seq_sim.csv',table.index)
    drug=read_static(root/'drug_str_sim.csv',table.columns)
    candidates=np.empty((0,2),dtype=np.int64)
    if args.case_pairs:
        requested=pd.read_csv(args.case_pairs)
        raw=requested[['circ_index','drug_index']].to_numpy(dtype=np.float64)
        if not np.isfinite(raw).all() or not np.equal(raw,np.floor(raw)).all():
            raise ValueError('Case indices must be integers')
        candidates=raw.astype(np.int64)
    for index in args.case_drug_indices:
        if not 0<=index<table.shape[1]: raise ValueError('Case drug index out of range')
        candidates=np.vstack([candidates,np.column_stack([np.arange(table.shape[0]),np.full(table.shape[0],index)])])
    return table,circ,drug,np.unique(candidates,axis=0)


def compute_features(training,circ_static,drug_static):
    A=training.A_train
    circ_gip,drug_gip=gip(A),gip(A.T)
    circ_entropy,drug_entropy=entropy_similarity(A),entropy_similarity(A.T)
    circ=np.where(circ_static>0,(circ_static+circ_gip+circ_entropy)/3,circ_gip)
    drug=np.where(drug_static>0,(drug_static+drug_gip+drug_entropy)/3,drug_gip)
    return dict(A=torch.tensor(A.copy(),dtype=torch.float32),
                reconstruction_mask=torch.tensor(training.train_mask.copy(),dtype=torch.bool),
                IM=torch.tensor(circ,dtype=torch.float32),ID=torch.tensor(drug,dtype=torch.float32),
                circ_gip=circ_gip,drug_gip=drug_gip,circ_entropy=circ_entropy,drug_entropy=drug_entropy)


def feature_fingerprints(features):
    return {name:array_hash(value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else value)
            for name,value in features.items()}


def label_invariance_audit(training,circ_static,drug_static,evaluation,first_features):
    # Keep all pair sets fixed. Relabel only an independent evaluation object.
    from radvl.data_protocol import EvaluationSet, readonly
    altered=EvaluationSet(evaluation.pairs,readonly(1-evaluation.labels))
    if len(altered.labels) and np.array_equal(altered.labels,evaluation.labels):
        raise AssertionError('Evaluation-label perturbation did not change the labels')
    second=compute_features(training,circ_static,drug_static)
    before,after=feature_fingerprints(first_features),feature_fingerprints(second)
    if before!=after: raise AssertionError('Train-only feature construction is not invariant')
    return dict(fold=training.fold,evaluation_label_hash_before=array_hash(evaluation.labels),
                evaluation_label_hash_after=array_hash(altered.labels),
                train_input_and_target_hash=array_hash(training.A_train),
                train_mask_hash=array_hash(training.train_mask),features_unchanged=True,
                scope='pure train-only feature builder; not a second optimization run',
                feature_fingerprints=before)


def build_graphs(features,k,device):
    # Reuse graph architecture only; no old data loader is called.
    from radvl import hypergraph as ConstructHW
    from radvl.training_isolation import pyg_similarity_graph
    A=features['A'].to(device)
    circ_similarity=features['IM'].numpy()
    drug_similarity=features['ID'].numpy()
    circ=np.hstack([features['A'].numpy(),circ_similarity])
    drug=np.hstack([features['A'].numpy().T,drug_similarity])
    return dict(A=A,AT=A.T,
                circ_features=torch.tensor(circ,dtype=torch.float32,device=device),
                drug_features=torch.tensor(drug,dtype=torch.float32,device=device),
                circ_knn=ConstructHW.constructHW_knn(circ,K_neigs=[k],is_probH=False).to(device),
                circ_kmeans=ConstructHW.constructHW_kmean(circ,clusters=[9]).to(device),
                drug_knn=ConstructHW.constructHW_knn(drug,K_neigs=[k],is_probH=False).to(device),
                drug_kmeans=ConstructHW.constructHW_kmean(drug,clusters=[9]).to(device),
                circ_graph=pyg_similarity_graph(circ_similarity,k),
                drug_graph=pyg_similarity_graph(drug_similarity,k))


def forward(model,graphs):
    return model(graphs['circ_features'],graphs['drug_features'],
                 graphs['circ_knn'],graphs['circ_kmeans'],graphs['drug_knn'],graphs['drug_kmeans'],
                 graphs['AT'],graphs['A'],graphs['drug_graph'],graphs['circ_graph'])

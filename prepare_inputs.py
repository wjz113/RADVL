"""Validate user-supplied matrices and create an audited RADVL evaluation partition."""
import argparse,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--split-seed',type=int,default=1234)
    a=p.parse_args()
    import numpy as np
    import pandas as pd
    from radvl.data_protocol import prepare_protocol
    frames={n:pd.read_csv(a.data_dir/n,index_col=0) for n in ('association.csv','gene_seq_sim.csv','drug_str_sim.csv')}
    A=frames['association.csv'];cs=frames['gene_seq_sim.csv'];ds=frames['drug_str_sim.csv']
    if cs.shape!=(len(A),len(A)) or ds.shape!=(A.shape[1],A.shape[1]):raise ValueError('Similarity dimensions do not match the association matrix.')
    if not all(np.isfinite(x.to_numpy(dtype=float)).all() for x in frames.values()):raise ValueError('Inputs contain non-finite values.')
    if not np.isin(A.to_numpy(),[0,1]).all():raise ValueError('Association entries must be binary.')
    if list(map(str,cs.index))!=list(map(str,A.index)) or list(map(str,cs.columns))!=list(map(str,A.index)):raise ValueError('CircRNA-side identifier order differs between matrices.')
    if list(map(str,ds.index))!=list(map(str,A.columns)) or list(map(str,ds.columns))!=list(map(str,A.columns)):raise ValueError('Drug identifier order differs between matrices.')
    target=ROOT/'protocols'/f'edge_val125_seed{a.split_seed}'
    protocol=prepare_protocol(target,a.data_dir/'association.csv',np.empty((0,2),dtype=np.int64),folds=5,split_seed=a.split_seed,scenario='edge',validation_fraction=.125)
    for fold in range(len(protocol)):protocol.audit(fold)
    print('Validated input shape:',A.shape,'positive associations:',int(A.to_numpy().sum()))
    print('Prepared and audited',len(protocol),'folds in',target)

if __name__=='__main__':main()

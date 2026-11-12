"""RADVL recorder protocol."""
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from radvl.recorder_isolation import IsolatedRecorder
from radvl.recorder_base import append_csv
from radvl.data_protocol import array_hash, file_hash


class ProtocolRecorder(IsolatedRecorder):
    def __init__(self,directory,args,k,dataset,protocol,requested_args):
        super().__init__(directory,args,k,dataset)
        self.protocol=protocol
        shutil.copytree(protocol.directory,self.directory/'shared_protocol')
        root=Path(__file__).resolve().parent
        extra=['training_protocol.py','settings_protocol.py',
               'data_protocol.py','feature_inputs.py',
               'recorder_protocol.py','protocol_adapter.py',
               'training_isolation.py','data_preprocessing.py',
               'settings_isolation.py']
        path=self.directory/'run_config.json'
        config=json.loads(path.read_text(encoding='utf-8'))
        for name in extra:
            shutil.copy2(root/name,self.directory/'source'/name)
            config['input_and_source_sha256']['source/'+name]=file_hash(root/name)
        provenance=None
        if requested_args.static_provenance:
            path=Path(requested_args.static_provenance)
            provenance=json.loads(path.read_text(encoding='utf-8'))
            shutil.copy2(path,self.directory/'input_data'/'static_provenance.json')
        config.update(protocol='shared_frozen_train_only_protocol',
                      requested_arguments=vars(requested_args),
                      protocol_manifest_sha256=file_hash(protocol.directory/'manifest.json'),
                      protocol_manifest=protocol.manifest,
                      feature_cache_policy='no cross-fold trained cache reuse',
                      model_prediction='0.9 * association + 0.1 * fused; unchanged, no MLP',
                      static_provenance=provenance,
                      static_provenance_status='user-supplied evidence; requires upstream review' if provenance else 'unverified association-independence assumption',
                      evaluation_label_isolation='fit accepts TrainingFold only; test labels exposed after fit',
                      notes=['Original architecture and model hyperparameters retained.',
                             'Data/split/negative ratios are shared experimental settings, not model hyperparameters.',
                             'No hyperparameter search, no early stopping in DGAA.',
                             'Optional internal validation is excluded from inputs/losses but not used to tune this model.',
                             'Classification threshold fixed at 0.5 for DGAA probabilities.',
                             'Fold seeds are base training seed + fit index.',
                             'Cold-both uses Cartesian blocks; k folds per axis requires k*k fits.',
                             'Passing structural audits is not proof of independent upstream static features.'])
        self.write_json('run_config.json',config)

    def fold_inputs(self,fold,inputs):
        audit=self.protocol.audit(fold)
        training=self.protocol.training(fold)
        if not np.array_equal(inputs['A'].numpy(),training.A_train):
            raise AssertionError('Actual input differs from frozen training matrix')
        if not np.array_equal(inputs['reconstruction_mask'].numpy(),training.train_mask):
            raise AssertionError('Actual loss mask differs from frozen training mask')
        append_csv(self.directory/'protocol_audit.csv',[audit])
        np.savez_compressed(self.directory/f'fold_preprocessing_{fold}.npz',
            **{name:value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else value
               for name,value in inputs.items()})
        records=[]
        for side,axis,names in [('circ',0,self.circ_names),('drug',1,self.drug_names)]:
            degree=training.A_train.sum(axis=1 if axis==0 else 0)
            for index,name in enumerate(names):
                records.append(dict(fold=fold,entity_type=side,index=index,identifier=name,
                                    training_positive_degree=int(degree[index]),input_positive_degree=int(degree[index])))
        append_csv(self.directory/'entity_degrees.csv',records)

    def graph_inputs(self,fold,*args):
        super().graph_inputs(fold,*args)
        training=self.protocol.training(fold)
        with np.load(self.directory/f'graph_inputs_{fold}.npz',allow_pickle=False) as stored:
            for name in stored.files:
                append_csv(self.directory/'feature_lineage.csv',[dict(
                    fold=fold,artifact=name,sha256=array_hash(stored[name]),
                    training_fingerprint=training.fingerprint,
                    source='this-fold A_train and association-independent static inputs only')])

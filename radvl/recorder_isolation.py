"""RADVL recorder isolation."""
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from radvl.recorder_metadata import OriginalRecorder
from radvl.recorder_base import RunRecorder, append_csv


class IsolatedRecorder(OriginalRecorder):
    def __init__(self, directory, args, k, dataset):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.args, self.dataset, self.splits = args, dataset, dataset['md']
        self.fold_metrics, self.compute_seconds, self.record_seconds = [], {}, {}
        self.started = time.perf_counter()
        self.association = dataset['md_true'].numpy()
        self.circ_names = dataset['association'].index.astype(str).tolist()
        self.drug_names = dataset['association'].columns.astype(str).tolist()
        self.oof = np.full(self.association.shape, np.nan, dtype=np.float64)
        self.oof_fold = np.full(self.association.shape, -1, dtype=np.int32)
        self.candidate_fold_scores = []
        source = self.directory / 'source'
        source.mkdir()
        root = Path(__file__).resolve().parent
        names = ['training_isolation.py', 'data_preprocessing.py',
                 'settings_isolation.py', 'recorder_isolation.py',
                 'model_components.py', 'regularization_utils.py',
                 'attention_layers.py', 'recorder_metadata.py',
                 'recorder_base.py', 'encoder_layers.py', 'hypergraph.py',
                 'hypergraph_neighbors.py', 'hypergraph_clustering.py']
        hashes = {}
        for name in names:
            shutil.copy2(root / name, source / name)
            hashes['source/'+name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        inputs = self.directory / 'input_data'
        inputs.mkdir()
        for name in ['association.csv', 'gene_seq_sim.csv', 'drug_str_sim.csv',
                     'gene_map.csv', 'drug_pubchem_map.csv']:
            path = Path(args.data_path) / name
            if path.exists():
                shutil.copy2(path, inputs / name)
                hashes['data/'+name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if args.case_pairs:
            path = Path(args.case_pairs)
            shutil.copy2(path, inputs / 'case_pairs_requested.csv')
            hashes['data/case_pairs_requested.csv'] = hashlib.sha256(path.read_bytes()).hexdigest()
        pd.DataFrame(dataset['candidates'], columns=['circ_index', 'drug_index']).to_csv(
            self.directory / 'case_pairs_resolved.csv', index=False)
        versions = {}
        for package in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch-geometric', 'torch-sparse']:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        self.write_json('run_config.json', dict(
            arguments=vars(args), argv=sys.argv, python=sys.version, executable=sys.executable,
            working_directory=str(Path.cwd()), torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            package_versions=versions, input_and_source_sha256=hashes,
            protocol='fold_local_A_reconstruction_training_pairs_only',
            training_seed=args.seed, split_seed=args.split_seed, kmeans_seed=0,
            classification_threshold=0.5, hyperparameter_search=False, early_stopping=False,
            optimizer=dict(name='AdamW', lr=1e-4, weight_decay=5e-4),
            model='original_record_architecture', hidden_list=[256,256], neighbors=k,
            num_hidden1=512, latent_dim=64, num_layers_argument=4, gin_layers=5,
            reconstruction='mean BCE over sampled training positive and negative pairs only',
            similarity_fusion='where static>0: (static+training_GIP+training_entropy)/3; else training_GIP',
            entropy='global degree-weighted overlap; zero denominator -> zero',
            static_provenance_assumption='gene_seq_sim and drug_str_sim are association-independent; verify upstream generation',
            candidate_policy='unknown queries excluded from all negative sampling; known queries scored only OOF',
            performance_note='synchronized recorded training; eval forward under no_grad; not a repeated inference benchmark',
            notes=['No combined/integration/RWR inputs are read.',
                   'Full association labels are used only to define splits/exclude known positives and evaluate.',
                   'Unknown labels remain presumed negatives, not biological confirmation.',
                   'Final checkpoints are not selected using test labels.',
                   'No inner validation tuning or cold-start claim.',
                   'Legacy test-max-F1 evaluation was replaced with fixed threshold 0.5.']))
        np.savez_compressed(self.directory / 'association_labels_evaluation_only.npz', association=self.association)
        for fold, split in enumerate(self.splits):
            np.savez_compressed(self.directory / f'split_{fold}.npz',
                train_positive=split['train'][0].numpy(), train_negative=split['train'][1].numpy(),
                test_positive=split['test'][0].numpy(), test_negative=split['test'][1].numpy())

    def fold_inputs(self, fold, inputs):
        split = self.splits[fold]
        train_pos, train_neg = [value.numpy() for value in split['train']]
        test_pos, test_neg = [value.numpy() for value in split['test']]
        test_pairs = np.vstack([test_pos, test_neg])
        A = inputs['A'].numpy()
        mask = inputs['reconstruction_mask'].numpy()
        candidates = self.dataset['candidates']
        unknown_candidates = candidates[self.association[candidates[:,0], candidates[:,1]] == 0]
        training_set = set(map(tuple, np.vstack([train_pos, train_neg])))
        testing_set = set(map(tuple, test_pairs))
        audit = dict(fold=fold, training_positive_count=len(train_pos), training_negative_count=len(train_neg),
                     test_positive_count=len(test_pos), test_negative_count=len(test_neg),
                     full_known_positive_count=int(self.association.sum()),
                     accounted_positive_count=len(train_pos)+len(test_pos),
                     train_test_pair_overlap=len(training_set & testing_set),
                     test_positive_in_A=int(A[test_pos[:,0],test_pos[:,1]].sum()),
                     test_pairs_in_reconstruction=int(mask[test_pairs[:,0],test_pairs[:,1]].sum()),
                     unknown_candidates_in_training_negative=len(set(map(tuple,train_neg)) & set(map(tuple,unknown_candidates))),
                     unknown_candidates_in_reconstruction=int(mask[unknown_candidates[:,0],unknown_candidates[:,1]].sum()),
                     GIP_entropy_source='this_fold_training_A_only')
        forbidden = ['train_test_pair_overlap', 'test_positive_in_A', 'test_pairs_in_reconstruction',
                     'unknown_candidates_in_training_negative', 'unknown_candidates_in_reconstruction']
        if any(audit[name] != 0 for name in forbidden):
            raise AssertionError(f'Fold isolation audit failed: {audit}')
        if audit['accounted_positive_count'] != audit['full_known_positive_count']:
            raise AssertionError('Positive fold coverage is incomplete')
        append_csv(self.directory / 'protocol_audit.csv', [audit])
        np.savez_compressed(self.directory / f'fold_preprocessing_{fold}.npz',
            **{name:value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else value
               for name,value in inputs.items()})
        rows=[]
        for side, axis, names in [('circ',0,self.circ_names),('drug',1,self.drug_names)]:
            degree=np.bincount(train_pos[:,axis],minlength=len(names))
            for index,name in enumerate(names):
                rows.append(dict(fold=fold,entity_type=side,index=index,identifier=name,
                                 training_positive_degree=int(degree[index]),
                                 input_positive_degree=int(A.sum(axis=1 if axis==0 else 0)[index])))
        append_csv(self.directory / 'entity_degrees.csv',rows)

    def save_fold(self, fold, model, optimizer, args, data, metrics):
        # Call base storage directly: no legacy test-threshold optimization.
        RunRecorder.save_fold(self, fold, model, optimizer, args, data, metrics)
        pairs = torch.cat(data[3],dim=0).numpy()
        predictions = np.load(self.directory / f'pred_{fold}.npy').reshape(-1)
        labels = np.load(self.directory / f'true_{fold}.npy').reshape(-1)
        idx=(pairs[:,0],pairs[:,1])
        if np.isfinite(self.oof[idx]).any():
            raise AssertionError('A pair was evaluated in more than one fold')
        self.oof[idx]=predictions
        self.oof_fold[idx]=fold
        tn,fp,fn,tp=confusion_matrix(labels,predictions>=0.5,labels=[0,1]).ravel()
        append_csv(self.directory / 'confusion_matrices.csv', [dict(
            fold=fold,threshold=0.5,TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn))])
        fpr,tpr,thresholds=roc_curve(labels,predictions)
        precision,recall,pr_thresholds=precision_recall_curve(labels,predictions)
        np.savez_compressed(self.directory / f'curves_{fold}.npz',fpr=fpr,tpr=tpr,
                            roc_thresholds=thresholds,precision=precision,recall=recall,pr_thresholds=pr_thresholds)
        np.savez_compressed(self.directory / f'all_scores_diagnostic_fold_{fold}.npz',
                            **{name:value.cpu().numpy() for name,value in model.diagnostic_scores.items()})
        np.savez_compressed(self.directory / f'embeddings_{fold}.npz',
            circ_mu=model.z_node_mean.detach().cpu().numpy(),drug_mu=model.z_edge_mean.detach().cpu().numpy(),
            circ_log_std=model.z_node_log_std.detach().cpu().numpy(),drug_log_std=model.z_edge_log_std.detach().cpu().numpy(),
            **{name+'_similarity':value.cpu().numpy() for name,value in model.diagnostic_embeddings.items()})
        np.savez_compressed(self.directory / f'fusion_details_{fold}.npz',
            **{side+'_'+name:value.cpu().numpy() for side,fusion in [('circ',model.circ_fusion),('drug',model.dr_fusion)]
               for name,value in fusion.diagnostic_arrays.items()})
        candidates=self.dataset['candidates']
        unknown=candidates[self.association[candidates[:,0],candidates[:,1]]==0]
        scores=model.diagnostic_scores['combined'].cpu().numpy()
        rows=[]
        for circ,drug in unknown:
            rows.append(dict(fold=fold,circ_index=int(circ),drug_index=int(drug),score=float(scores[circ,drug]),
                             score_source='reserved_unknown_not_training_negative'))
        self.candidate_fold_scores.extend(rows)
        append_csv(self.directory / 'unknown_candidate_fold_scores.csv',rows)

    def finish(self):
        known=np.argwhere(self.association==1)
        if not np.isfinite(self.oof[known[:,0],known[:,1]]).all():
            raise AssertionError('Missing OOF score for a known edge')
        np.savez_compressed(self.directory / 'oof_scores.npz',scores=self.oof,fold_assignment=self.oof_fold)
        # test_predictions already contains exactly one held-out score per evaluated pair.
        test_table=pd.read_csv(self.directory / 'test_predictions.csv')
        test_table.to_csv(self.directory / 'oof_predictions.csv',index=False)
        test_table[test_table.label==1].to_csv(self.directory / 'known_edges_oof.csv',index=False)
        candidates=self.dataset['candidates']
        rows=[]
        candidate_values={}
        for row in self.candidate_fold_scores:
            candidate_values.setdefault((row['circ_index'],row['drug_index']),[]).append(row['score'])
        for circ,drug in candidates:
            is_known=bool(self.association[circ,drug])
            if is_known:
                score=float(self.oof[circ,drug]); fold=int(self.oof_fold[circ,drug]); source='known_edge_single_oof'
                std=None; fits=1
            else:
                values=candidate_values.get((int(circ),int(drug)),[])
                if len(values)!=self.args.validation:
                    raise AssertionError('Unknown case candidate missing held-out fold prediction')
                score=float(np.mean(values)); std=float(np.std(values)); fits=len(values)
                fold=None; source='reserved_unknown_crossfit_mean'
            rows.append(dict(circ_index=int(circ),drug_index=int(drug),circ_id=self.circ_names[circ],
                             drug_id=self.drug_names[drug],is_known=is_known,score=score,
                             score_source=source,heldout_fold=fold,fit_count=fits,score_std=std))
        if rows:
            table=pd.DataFrame(rows)
            # Ranking candidates uses unknown pairs only, not previously known edges.
            table['unknown_rank_within_drug']=np.nan
            unknown=~table.is_known
            table.loc[unknown,'unknown_rank_within_drug']=table.loc[unknown].groupby('drug_index').score.rank(
                method='first',ascending=False)
            table.to_csv(self.directory / 'case_candidate_scores.csv',index=False)
        super().finish()

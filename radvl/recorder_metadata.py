"""RADVL recorder metadata."""
import importlib.metadata
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, precision_recall_curve
from radvl.recorder_base import RunRecorder, append_csv


def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def rss_megabytes():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 2
    except ImportError:
        return None


class OriginalRecorder(RunRecorder):
    def __init__(self, directory, args, k, dataset):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.args, self.splits, self.fold_metrics = args, dataset['md'], []
        self.started = time.perf_counter()
        self.compute_seconds, self.record_seconds = {}, {}
        association = pd.read_csv(Path(args.data_path) / 'association.csv', index_col=0)
        self.circ_names = association.index.astype(str).tolist()
        self.drug_names = association.columns.astype(str).tolist()
        self.association = association.values
        source = self.directory / 'source'
        source.mkdir()
        root = Path(__file__).resolve().parent
        names = ['training_recording.py', 'model_components.py',
                 'settings_recording.py', 'regularization_utils.py',
                 'attention_layers.py', 'recorder_metadata.py',
                 'recorder_base.py', 'data_loading.py', 'data_dataset.py',
                 'encoder_layers.py', 'hypergraph.py', 'hypergraph_neighbors.py',
                 'hypergraph_clustering.py', 'metrics.py']
        import hashlib
        hashes = {}
        for name in names:
            shutil.copy2(root / name, source / name)
            hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        data_directory = self.directory / 'input_data'
        data_directory.mkdir()
        for name in ['association.csv', 'drug_combined_sim.csv', 'gene_combined_sim.csv',
                     'integration_drug.csv', 'integration_circRNA.csv',
                     'gene_map.csv', 'drug_pubchem_map.csv']:
            path = Path(args.data_path) / name
            if path.exists():
                shutil.copy2(path, data_directory / name)
                hashes[f'data/{name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
        versions = {}
        for package in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch-geometric',
                        'torch-sparse', 'networkx', 'matplotlib']:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        config = dict(arguments=vars(args), argv=sys.argv, working_directory=str(Path.cwd()),
                      python=sys.version, executable=sys.executable,
                      torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      package_versions=versions, input_and_source_sha256=hashes,
                      training_seed=args.seed, split_seed=114514, kmeans_seed=0,
                      eval_seeds=list(range(10)), eval_seeds_are_training_repeats=False,
                      protocol='legacy_edge_cv_unmasked', run_label=args.run_label,
                      index_base=0, optimizer=dict(name='AdamW', lr=1e-4, weight_decay=5e-4),
                      architecture=dict(hidden_list=[256, 256], num_hidden1=512,
                                        latent_dim=64, num_layers_argument=4, gin_layers=5,
                                        neighbors=k, kmeans_clusters=9),
                      loss_weights=dict(mse_positive=1-args.alpha, mse_negative=args.alpha,
                                        explicit_l2=1e-4, reconstruction_bce=1,
                                        original_kl=0.5, covariance=args.co_lambda, alignment=2),
                      notes=['Original architecture, loss, data split and evaluation retained.',
                             'Test positives remain in features and reconstruction targets.',
                             'Not a leakage-free protocol or an actual cold-start experiment.',
                             'Labels 0 denote unlabeled presumed negatives, not verified negatives.',
                             'Final checkpoint only; no validation selection or early stopping.',
                             'run_label is metadata, not an ablation switch.',
                             'No confidence interval or significance test is inferred from five dependent folds.',
                             'CUDA timing synchronizes; logging overhead reported separately.',
                             'Inference timing is one eval forward with autograd enabled, prebuilt graphs; not a benchmark.'])
        self.write_json('run_config.json', config)
        np.savez_compressed(self.directory / 'association_labels.npz', association=self.association)
        audit = []
        for fold, split in enumerate(self.splits):
            arrays = dict(train_positive=split['train'][0].cpu().numpy(),
                          train_negative=split['train'][1].cpu().numpy(),
                          test_positive=split['test'][0].cpu().numpy(),
                          test_negative=split['test'][1].cpu().numpy())
            np.savez_compressed(self.directory / f'split_{fold}.npz', **arrays)
            tp = arrays['test_positive']
            training = set(map(tuple, np.vstack([arrays['train_positive'], arrays['train_negative']])))
            testing = set(map(tuple, np.vstack([tp, arrays['test_negative']])))
            audit.append(dict(fold=fold, **{name+'_count': len(value) for name, value in arrays.items()},
                              train_test_pair_overlap=len(training & testing),
                              test_positive_present_in_input=int((self.association[tp[:, 0], tp[:, 1]] >= 1).sum()),
                              test_positive_present_in_bce_target=int((self.association[tp[:, 0], tp[:, 1]] >= 1).sum()),
                              observed_full_positive_count=int((self.association >= 1).sum()),
                              used_positive_count=len(arrays['train_positive']) + len(tp)))
            training_positive = arrays['train_positive']
            entities = []
            for entity_type, axis, identifiers in [('circ', 0, self.circ_names), ('drug', 1, self.drug_names)]:
                training_degree = np.bincount(training_positive[:, axis], minlength=len(identifiers))
                input_degree = (self.association >= 1).sum(axis=1 if axis == 0 else 0)
                for index, identifier in enumerate(identifiers):
                    entities.append(dict(fold=fold, entity_type=entity_type, index=index,
                                         identifier=identifier, training_positive_degree=int(training_degree[index]),
                                         input_positive_degree=int(input_degree[index])))
            append_csv(self.directory / 'entity_degrees.csv', entities)
        append_csv(self.directory / 'protocol_audit.csv', audit)

    def write_json(self, name, value):
        with (self.directory / name).open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)

    def epoch(self, fold, epoch, total, losses, model, optimizer, compute_seconds):
        started = time.perf_counter()
        self.compute_seconds[fold] = self.compute_seconds.get(fold, 0) + compute_seconds
        row = dict(fold=fold, seed=self.args.seed, epoch=epoch,
                   total_loss=total.detach().item(), compute_seconds=compute_seconds,
                   learning_rate=optimizer.param_groups[0]['lr'])
        row.update({name: value.detach().item() for name, value in losses.items()})
        for side, mu, log_std, fusion in [
            ('circ', model.z_node_mean, model.z_node_log_std, model.circ_fusion),
            ('drug', model.z_edge_mean, model.z_edge_log_std, model.dr_fusion)]:
            row.update({f'{side}_mu_abs_mean': mu.detach().abs().mean().item(),
                        f'{side}_mu_std': mu.detach().std(unbiased=False).item(),
                        f'{side}_posterior_variance_mean': (2 * log_std.detach()).exp().mean().item()})
            row.update({f'{side}_{key}': value for key, value in fusion.diagnostics.items()})
        for name, module in [('all', model), ('circ_vae', model.node_encoders1),
                             ('drug_vae', model.hyperedge_encoders1),
                             ('circ_fusion', model.circ_fusion), ('drug_fusion', model.dr_fusion),
                             ('circ_gin', model.circ_gin_encoder), ('drug_gin', model.dr_gin_encoder),
                             ('circ_hgcn', model.hgcn_circ), ('drug_hgcn', model.hgcn_dr)]:
            row[name+'_gradient_norm'] = sum(p.grad.detach().float().square().sum().item()
                for p in module.parameters() if p.grad is not None) ** 0.5
        for name, scores in model.diagnostic_scores.items():
            for kind, pairs in zip(['positive', 'negative'], self.splits[fold]['train']):
                pairs = pairs.to(scores.device)
                selected = scores[pairs[:, 0], pairs[:, 1]]
                row[f'train_{name}_{kind}_mean'] = selected.mean().item()
                row[f'train_{name}_{kind}_std'] = selected.std(unbiased=False).item()
        row['gpu_peak_allocated_mb'] = (torch.cuda.max_memory_allocated()/1024**2
                                        if torch.cuda.is_available() else None)
        row['process_rss_mb'] = rss_megabytes()
        row['all_recorded_values_finite'] = bool(all(np.isfinite(value) for value in row.values()
            if isinstance(value, (int, float))))
        append_csv(self.directory / 'training_history.csv', [row])
        self.record_seconds[fold] = self.record_seconds.get(fold, 0) + time.perf_counter()-started

    def graph_inputs(self, fold, A, circ_features, drug_features,
                     circ_knn, circ_kmeans, drug_knn, drug_kmeans,
                     circ_graph, drug_graph):
        tensors = dict(association_input=A, circ_features=circ_features, drug_features=drug_features,
                       circ_knn_operator=circ_knn, circ_kmeans_operator=circ_kmeans,
                       drug_knn_operator=drug_knn, drug_kmeans_operator=drug_kmeans,
                       circ_graph_x=circ_graph.x, circ_edge_index=circ_graph.edge_index,
                       drug_graph_x=drug_graph.x, drug_edge_index=drug_graph.edge_index)
        np.savez_compressed(self.directory / f'graph_inputs_{fold}.npz',
                            **{key: value.detach().cpu().numpy() for key, value in tensors.items()})

    def save_fold(self, fold, model, optimizer, args, data, metrics):
        super().save_fold(fold, model, optimizer, args, data, metrics)
        np.savez_compressed(self.directory / f'full_scores_{fold}.npz',
                            **{name: value.cpu().numpy() for name, value in model.diagnostic_scores.items()})
        np.savez_compressed(self.directory / f'embeddings_{fold}.npz',
                            circ_mu=model.z_node_mean.detach().cpu().numpy(),
                            drug_mu=model.z_edge_mean.detach().cpu().numpy(),
                            circ_log_std=model.z_node_log_std.detach().cpu().numpy(),
                            drug_log_std=model.z_edge_log_std.detach().cpu().numpy(),
                            **{name+'_similarity': value.cpu().numpy()
                               for name, value in model.diagnostic_embeddings.items()})
        np.savez_compressed(self.directory / f'fusion_details_{fold}.npz',
                            **{f'{side}_{name}': value.cpu().numpy()
                               for side, fusion in [('circ', model.circ_fusion), ('drug', model.dr_fusion)]
                               for name, value in fusion.diagnostic_arrays.items()})
        labels = np.load(self.directory / f'true_{fold}.npy').reshape(-1)
        scores = np.load(self.directory / f'pred_{fold}.npy').reshape(-1)
        thresholds = np.unique(scores)
        thresholds = thresholds[(len(thresholds)*np.arange(1,1000)/1000).astype(int)]
        predicted = scores[None, :] >= thresholds[:, None]
        tp = (predicted & (labels[None, :] == 1)).sum(1)
        fp = (predicted & (labels[None, :] == 0)).sum(1)
        fn = int((labels == 1).sum())-tp
        best = np.argmax(2*tp / np.maximum(2*tp+fp+fn, 1))
        rows = []
        for method, threshold in [('legacy_test_max_f1', float(thresholds[best])), ('fixed_0.5', 0.5)]:
            prediction = scores >= threshold
            true_pos = int(((labels == 1) & prediction).sum())
            false_pos = int(((labels == 0) & prediction).sum())
            false_neg = int(((labels == 1) & ~prediction).sum())
            true_neg = int(((labels == 0) & ~prediction).sum())
            recall = true_pos/max(true_pos+false_neg,1)
            specificity = true_neg/max(true_neg+false_pos,1)
            rows.append(dict(fold=fold, method=method, threshold=threshold,
                             TP=true_pos, FP=false_pos, FN=false_neg, TN=true_neg,
                             Accuracy=(true_pos+true_neg)/len(labels), Recall=recall,
                             Specificity=specificity, Precision=true_pos/max(true_pos+false_pos,1),
                             F1=2*true_pos/max(2*true_pos+false_pos+false_neg,1),
                             balanced_accuracy_identity_error=(true_pos+true_neg)/len(labels)-(recall+specificity)/2))
        append_csv(self.directory / 'confusion_matrices.csv', rows)
        legacy = rows[0]
        metric_values = np.asarray(metrics).reshape(-1)
        append_csv(self.directory / 'metric_consistency.csv', [dict(
            fold=fold,
            reported_balanced_accuracy_identity_error=float(metric_values[3]-(metric_values[4]+metric_values[5])/2),
            reconstructed_minus_reported_f1=legacy['F1']-float(metric_values[2]),
            reconstructed_minus_reported_accuracy=legacy['Accuracy']-float(metric_values[3]),
            note='Legacy threshold reconstructed from 999 quantile candidates; float tie handling may differ')])
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        np.savez_compressed(self.directory / f'curves_{fold}.npz',
                            fpr=fpr, tpr=tpr, roc_thresholds=roc_thresholds,
                            precision=precision, recall=recall, pr_thresholds=pr_thresholds)

    def performance(self, fold, model, train_seconds, inference_seconds):
        append_csv(self.directory / 'performance.csv', [dict(
            fold=fold, seed=self.args.seed,
            parameter_count=sum(p.numel() for p in model.parameters()),
            trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
            parameters_with_gradient_last_step=sum(p.numel() for p in model.parameters() if p.grad is not None),
            train_wall_seconds=train_seconds, epoch_compute_seconds=self.compute_seconds.get(fold, 0),
            epoch_record_seconds=self.record_seconds.get(fold, 0),
            eval_forward_seconds=inference_seconds,
            gpu_peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None,
            gpu_peak_reserved_mb=torch.cuda.max_memory_reserved()/1024**2 if torch.cuda.is_available() else None,
            process_rss_mb=rss_megabytes())])

    def finish(self):
        super().finish()
        self.write_json('completion.json', dict(status='complete', folds=len(self.fold_metrics),
                                               wall_seconds_since_recorder=time.perf_counter()-self.started))

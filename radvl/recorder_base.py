"""RADVL recorder base."""
import csv
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


METRICS = ['AUC', 'AUPR', 'F1', 'Accuracy', 'Recall', 'Specificity', 'Precision']


def append_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


class RunRecorder:
    def __init__(self, directory, args, k, dataset):
        self.args = args
        self.splits = dataset['md']
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fold_metrics = []
        association = pd.read_csv(Path(args.data_path) / 'association.csv', index_col=0)
        self.circ_names = association.index.astype(str).tolist()
        self.drug_names = association.columns.astype(str).tolist()
        source_directory = Path(__file__).resolve().parent
        snapshots = self.directory / 'source'
        snapshots.mkdir()
        source_files = [
            'training_reference.py', 'model_reference.py',
            'settings_reference.py', 'objectives_reference.py',
            'recorder_base.py', 'data_loading.py', 'data_dataset.py',
            'metrics.py', 'hypergraph.py',
            'hypergraph_neighbors.py', 'hypergraph_clustering.py',
        ]
        for name in source_files:
            shutil.copy2(source_directory / name, snapshots / name)
        hashes = {}
        for name in ['association.csv', 'drug_combined_sim.csv',
                     'gene_combined_sim.csv', 'integration_drug.csv',
                     'integration_circRNA.csv']:
            path = Path(args.data_path) / name
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        config = dict(
            arguments=vars(args), command=sys.argv, working_directory=str(Path.cwd()),
            python=sys.version, torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda, cuda_available=torch.cuda.is_available(),
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            numpy_version=np.__version__, neighbors=k,
            architecture=dict(hidden_list=[256, 256], num_proj_hidden=64,
                              num_hidden1=512, num_out=64, num_layers=4,
                              gin_layers=5),
            optimizer=dict(name='AdamW', lr=0.0001, weight_decay=5e-4),
            seeds=dict(model_import=1234, data_split=114514),
            input_sha256=hashes, index_base=0,
            limitations=[
                'Association inputs remain unmasked; this run is not leakage-free.',
                'Legacy split, graph construction, score mixture and metric rules retained.',
                'Classification threshold maximizes F1 on the test labels.',
                'Checkpoint is final epoch, not selected using test performance.',
                'Association CSV headers are identifiers; biological names are not inferred.',
            ])
        with (self.directory / 'run_config.json').open('w', encoding='utf-8') as stream:
            json.dump(config, stream, indent=2, ensure_ascii=False)
        for fold, split in enumerate(dataset['md']):
            np.savez_compressed(
                self.directory / f'split_{fold}.npz',
                train_positive=split['train'][0].cpu().numpy(),
                train_negative=split['train'][1].cpu().numpy(),
                test_positive=split['test'][0].cpu().numpy(),
                test_negative=split['test'][1].cpu().numpy())

    def epoch(self, fold, epoch, total, parts, model, optimizer):
        row = dict(fold=fold, epoch=epoch, total_loss=total.detach().item())
        row.update({name: value.detach().item() for name, value in parts.items()})
        row.update(
            weighted_av=self.args.lambda_av * row['association_view'],
            weighted_cov=self.args.lambda_av * self.args.co_lambda * row['covariance'],
            weighted_alignment=self.args.lambda_align * row['alignment'],
            learning_rate=optimizer.param_groups[0]['lr'])
        for side, mu, logvar, fusion in [
            ('circ', model.z_node_mean, model.z_node_log_var, model.circ_fusion),
            ('drug', model.z_edge_mean, model.z_edge_log_var, model.dr_fusion),
        ]:
            row.update({f'{side}_mu_abs_mean': mu.detach().abs().mean().item(),
                        f'{side}_mu_std': mu.detach().std(unbiased=False).item(),
                        f'{side}_posterior_variance_mean': logvar.detach().exp().mean().item()})
            row.update({f'{side}_{name}': value for name, value in fusion.diagnostics.items()})
        gradient_square = sum(p.grad.detach().float().square().sum().item()
                              for p in model.parameters() if p.grad is not None)
        row['gradient_norm'] = gradient_square ** 0.5
        for name in ['circ_association_encoder', 'dr_association_encoder',
                     'circ_fusion', 'dr_fusion', 'circ_gin_encoder',
                     'dr_gin_encoder', 'hgcn_circ', 'hgcn_dr']:
            square = sum(p.grad.detach().float().square().sum().item()
                         for p in getattr(model, name).parameters() if p.grad is not None)
            row[f'{name}_gradient_norm'] = square ** 0.5
        for name, scores in model.diagnostic_scores.items():
            for kind, pairs in zip(['positive', 'negative'], self.splits[fold]['train']):
                pairs = pairs.to(scores.device)
                selected = scores[pairs[:, 0], pairs[:, 1]]
                row[f'train_{name}_{kind}_mean'] = selected.mean().item()
                row[f'train_{name}_{kind}_std'] = selected.std(unbiased=False).item()
        row['all_recorded_values_finite'] = bool(all(np.isfinite(value)
            for value in row.values() if isinstance(value, (int, float))))
        append_csv(self.directory / 'training_history.csv', [row])

    def save_fold(self, fold, model, optimizer, args, data, metrics):
        values = np.asarray(metrics).reshape(-1)
        row = dict(fold=fold, **dict(zip(METRICS, values.tolist())))
        self.fold_metrics.append(row)
        append_csv(self.directory / 'fold_metrics.csv', [row])
        pairs = torch.cat(data[3], dim=0).cpu().numpy()
        labels = np.load(self.directory / f'true_{fold}.npy').reshape(-1)
        predictions = np.load(self.directory / f'pred_{fold}.npy').reshape(-1)
        if not (len(pairs) == len(labels) == len(predictions)):
            raise ValueError('Test pair indices, labels and predictions have different lengths')
        rows = []
        branches = {name: scores.detach().cpu().numpy()
                    for name, scores in model.diagnostic_scores.items()}
        for (circ, drug), label, prediction in zip(pairs, labels, predictions):
            item = dict(fold=fold, circ_index=int(circ), drug_index=int(drug),
                        circ_id=self.circ_names[circ], drug_id=self.drug_names[drug],
                        label=float(label), prediction=float(prediction))
            item.update({f'{name}_score': float(scores[circ, drug])
                         for name, scores in branches.items()})
            rows.append(item)
        append_csv(self.directory / 'test_predictions.csv', rows)
        branch_rows = []
        for name, matrix in branches.items():
            scores = matrix[pairs[:, 0], pairs[:, 1]]
            precision, recall, _ = precision_recall_curve(labels, scores)
            branch_rows.append(dict(
                fold=fold, branch=name, AUC=roc_auc_score(labels, scores),
                AUPR=auc(recall, precision), score_mean=float(scores.mean()),
                score_std=float(scores.std()), score_min=float(scores.min()),
                score_max=float(scores.max()),
                positive_score_mean=float(scores[labels == 1].mean()),
                negative_score_mean=float(scores[labels == 0].mean())))
        append_csv(self.directory / 'branch_metrics.csv', branch_rows)
        if getattr(args, 'save_checkpoints', 'on') == 'on':
            torch.save(dict(
                fold=fold, epoch=args.epoch, model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(), arguments=vars(args),
                metrics=row, numpy_rng_state=np.random.get_state(),
                python_rng_state=random.getstate(), torch_rng_state=torch.get_rng_state(),
                cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
                self.directory / f'model_fold_{fold}_final.pt')

    def finish(self):
        values = np.array([[row[name] for name in METRICS] for row in self.fold_metrics])
        append_csv(self.directory / 'summary_metrics.csv', [
            dict(statistic='mean', **dict(zip(METRICS, values.mean(axis=0).tolist()))),
            dict(statistic='std_population', **dict(zip(METRICS, values.std(axis=0).tolist()))),
        ])

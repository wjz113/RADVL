"""RADVL training weighted."""
import copy
import datetime
from pathlib import Path
import time

import numpy as np
import torch

from radvl.settings_similarity import parameter_parser
from radvl.data_protocol import prepare_protocol
from radvl.features_similarity import load_external_inputs, compute_features, label_invariance_audit, build_graphs
from radvl.protocol_adapter import ranking_metrics
from radvl.recorder_similarity import RADVLWeightedRecorder
from radvl.recorder_metadata import sync_device
from radvl.recorder_base import append_csv, METRICS
from radvl.model_similarity import RADVL
from radvl.optimization import reset_training_seed, preserved_rng
from radvl.training_core import fit, predict, initial_state_hash


RESULTS_ROOT = 'Results_radvl_aligned_weighted'


def main(args):
    table, circ_static, drug_static, candidates = load_external_inputs(args)
    protocol = prepare_protocol(args.protocol_dir, Path(args.data_path) / 'association.csv', candidates,
                                folds=args.validation, split_seed=args.split_seed, scenario=args.scenario,
                                train_fraction=args.train_fraction,
                                train_negative_ratio=args.train_negative_ratio,
                                test_negative_ratio=args.test_negative_ratio,
                                validation_fraction=args.validation_fraction)
    print('Frozen protocol:', protocol.directory, 'fits:', len(protocol), flush=True)
    if args.prepare_only:
        print('Protocol created/verified. No training requested.', flush=True)
        return
    print(f'Graph and fusion policy: gin_weighting={args.gin_weighting} '
          f'(normalization={args.gin_weight_normalization}); hyperedge_weight={args.hyperedge_weight} '
          f'(rescale={args.hyperedge_weight_rescale}); fusion_tau_mode={args.fusion_tau_mode} '
          f'(tau={args.fusion_tau}, tau_scale={args.fusion_tau_scale}).', flush=True)
    recorded_args = copy.copy(args)
    recorded_args.validation = len(protocol)
    dataset = dict(md=[], md_true=torch.tensor(table.values, dtype=torch.float32), association=table,
                   candidates=protocol.candidates, folds=[None] * len(protocol))
    for fold in range(len(protocol)):
        values = protocol.arrays(fold)
        dataset['md'].append(dict(
            train=[torch.from_numpy(values['train_positive']), torch.from_numpy(values['train_negative'])],
            validation=[torch.from_numpy(values['validation_positive']), torch.from_numpy(values['validation_negative'])],
            test=[torch.from_numpy(values['test_positive']), torch.from_numpy(values['test_negative'])]))
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory = Path(RESULTS_ROOT) / args.scenario / timestamp
    recorder = RADVLWeightedRecorder(directory, recorded_args, args.neighbors, dataset, protocol, args)
    print('Results:', directory.resolve(), flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold in range(len(protocol)):
        training = protocol.training(fold)
        reset_training_seed(args.seed)
        features = compute_features(training, circ_static, drug_static)
        invariant = label_invariance_audit(training, circ_static, drug_static,
                                           protocol.evaluation(fold, 'test'), features)
        recorder.write_json(f'label_invariance_{fold}.json', invariant)
        recorder.fold_inputs(fold, features)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        graphs = build_graphs(features, args.neighbors, device, args.clusters,
                              args.hyperedge_weight, args.hyperedge_weight_rescale == 'on')
        recorder.graph_inputs(fold, graphs)
        fold_seed = args.seed
        reset_training_seed(fold_seed)
        model = RADVL(n_circ=table.shape[0], n_drug=table.shape[1], args=args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
        if optimizer.state:
            raise AssertionError('Optimizer was not freshly initialized')
        append_csv(directory / 'fold_initialization.csv', [dict(
            fold=fold, training_seed=fold_seed, initial_model_sha256=initial_state_hash(model),
            optimizer_state_entries=0, training_fingerprint=training.fingerprint,
            cache_policy='fresh graphs, model, optimizer')])
        validation = protocol.evaluation(fold, 'validation')
        train_seconds = fit(model, optimizer, training, validation, graphs, recorded_args, recorder)
        sync_device()
        inference_started = time.perf_counter()
        with preserved_rng():
            scores = predict(model, graphs)
        sync_device()
        inference_seconds = time.perf_counter() - inference_started
        # Test labels are consumed only here, after optimization has finished.
        test = protocol.evaluation(fold, 'test')
        scores_numpy = scores.detach().cpu().numpy()
        predictions = scores_numpy[test.pairs[:, 0], test.pairs[:, 1]]
        values = ranking_metrics(test.labels, predictions, threshold=0.5)
        metrics = np.array([[values[name] for name in METRICS]])
        np.save(directory / f'true_{fold}.npy', test.labels)
        np.save(directory / f'pred_{fold}.npy', predictions)
        split = dataset['md'][fold]
        recording_view = (features['ID'], features['IM'], split['train'], split['test'],
                          features['A'], dataset['md_true'], features['A'], features['reconstruction_mask'])
        recorder.performance(fold, model, train_seconds, inference_seconds)
        recorder.save_fold(fold, model, optimizer, recorded_args, recording_view, metrics)
        print(f"Fold {fold+1}/{len(protocol)} TEST (weighted graphs, best validation-AUC checkpoint, "
              f"seed {fold_seed}): " + ' | '.join(f'{name}={values[name]:.6f}' for name in METRICS), flush=True)
        del scores, model, optimizer, graphs, features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    recorder.finish()
    summary = np.array([[row[name] for name in METRICS] for row in recorder.fold_metrics])
    print('TEST MEAN: ' + ' | '.join(f'{name}={summary[:, i].mean():.6f}' for i, name in enumerate(METRICS)), flush=True)
    print('TEST STD (population): ' + ' | '.join(f'{name}={summary[:, i].std():.6f}' for i, name in enumerate(METRICS)), flush=True)
    print('Results saved:', directory.resolve(), flush=True)


if __name__ == '__main__':
    main(parameter_parser())

"""RADVL training calibration."""
import argparse
import math
import sys
import time

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from radvl import training_core as radvl_base
from radvl import training_negative_sampling as negsample
from radvl.training_core import cpu_snapshot, initial_state_hash
from radvl.features import forward
from radvl.protocol_adapter import ranking_metrics
from radvl.recorder_metadata import sync_device
from radvl.recorder_base import append_csv, METRICS
from radvl.optimization import preserved_rng, capture_rng, restore_rng, AucEarlyStopping, EpochProgress

controls = negsample.controls
_WEIGHTED = negsample._WEIGHTED
_WEIGHTED.RESULTS_ROOT = 'Results_radvl_aligned_residual_negsample_v2'


class CalibratedResampler(negsample.ResampledNegativeObjective):
    """v1 resampler plus degree-weighted hard negatives and the training-time logit adjustment."""

    def __init__(self, base_objective):
        super().__init__(base_objective)
        self.hard_fraction = 0.0
        self.hard_alpha = 0.5
        self.logit_adjust = False
        self.probability = None

    def bind(self, training, seed):
        super().bind(training, seed)
        A = np.asarray(training.A_train)
        weight = ((A.sum(1)[self.pool[:, 0]] + 1) * (A.sum(0)[self.pool[:, 1]] + 1)) ** self.hard_alpha
        uniform = np.full(len(self.pool), 1.0 / len(self.pool))
        mixed = (1 - self.hard_fraction) * uniform + self.hard_fraction * weight / weight.sum()
        self.probability = mixed / mixed.sum()

    def __call__(self, model, reconstruction_logits, positive_pairs, negative_pairs, covariance,
                 alignment, args):
        if self.mode in ('epoch', 'hard'):
            count = min(len(self.pool), int(round(self.ratio * len(positive_pairs))))
            probability = self.probability if self.mode == 'hard' else None
            chosen = self.generator.choice(len(self.pool), size=count, replace=False, p=probability)
            negative_pairs = torch.as_tensor(self.pool[chosen], dtype=torch.long,
                                             device=positive_pairs.device)
        if not self.logit_adjust:
            return self.base_objective(model, reconstruction_logits, positive_pairs, negative_pairs,
                                       covariance, alignment, args)
        raw = model.prediction_logits
        model.prediction_logits = raw - math.log(len(negative_pairs) / len(positive_pairs))
        try:
            return self.base_objective(model, reconstruction_logits, positive_pairs, negative_pairs,
                                       covariance, alignment, args)
        finally:
            model.prediction_logits = raw


_SAMPLER = CalibratedResampler(negsample._SAMPLER.base_objective)
controls._CONTROLS_OBJECTIVE = _SAMPLER
SETTINGS = dict(edge_dropout=0.0, ema_decay=0.0, profile_smoothing=0.0, profile_neighbors=8)
STATE = dict(directory=None)


class ParameterEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for name, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)
            else:
                self.shadow[name].copy_(value)

    def swap(self, model):
        """Exchange the live weights with the averaged ones (call twice to swap back)."""
        live = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        self.shadow = live


def knn_row_operator(similarity, k):
    """Row-normalised top-k similarity operator without self loops."""
    similarity = similarity.clone().fill_diagonal_(float('-inf'))
    values, indices = similarity.topk(k, dim=1)
    values = values.clamp_min(0)
    operator = torch.zeros_like(similarity).scatter_(1, indices, values)
    return operator / operator.sum(1, keepdim=True).clamp_min(1e-12)


def smoothed_profiles(A, circ_similarity, drug_similarity, k, weight):
    """Encoder input ``A + w * max(S_c A, A S_d^T)``, using training-only A and similarities.

    Known entries stay 1 (the max with A keeps them), unknown entries receive the neighbour
    evidence.  Only the CoVAE encoders consume ``graphs['A']``; labels are the frozen pair lists.
    """
    circ_operator = knn_row_operator(circ_similarity.float(), k)
    drug_operator = knn_row_operator(drug_similarity.float(), k)
    neighbour = torch.maximum(circ_operator @ A, A @ drug_operator.T)
    return torch.maximum(A, weight * neighbour)


def validation_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def fit(model, optimizer, training, validation, graphs, args, recorder):
    """The shared ``fit`` loop plus edge dropout, EMA and a validation-chosen threshold."""
    if len(validation.labels) == 0 or len(np.unique(validation.labels)) != 2:
        raise ValueError('AUC early stopping requires positive and negative validation pairs')
    _SAMPLER.bind(training, args.seed)
    STATE['directory'] = recorder.directory
    if SETTINGS['profile_smoothing'] > 0:
        # In place, so the test forward in the shared loop sees the same encoder input.
        graphs['A'] = smoothed_profiles(graphs['A'], graphs['circ_features'], graphs['drug_features'],
                                        SETTINGS['profile_neighbors'], SETTINGS['profile_smoothing'])
    device = graphs['A'].device
    positive_pairs = torch.tensor(training.positive_pairs.copy(), dtype=torch.long, device=device)
    negative_pairs = torch.tensor(training.negative_pairs.copy(), dtype=torch.long, device=device)
    stopping = AucEarlyStopping(args.patience, args.min_delta)
    progress = EpochProgress(training.fold, args.validation, args.epoch)
    ema = ParameterEMA(model, SETTINGS['ema_decay']) if SETTINGS['ema_decay'] > 0 else None
    dropout_rng = torch.Generator(device=device)
    dropout_rng.manual_seed(int(args.seed) * 1009 + int(training.fold))
    best_state, best_validation_aupr, best_scores = None, None, None
    validation_seconds, stopped_epoch = 0.0, 0
    sync_device(); started = time.perf_counter()
    try:
        for epoch in range(1, args.epoch + 1):
            model.train()
            sync_device(); epoch_started = time.perf_counter()
            train_graphs = graphs
            if SETTINGS['edge_dropout'] > 0:
                keep = torch.rand(graphs['A'].shape, generator=dropout_rng, device=device) >= SETTINGS['edge_dropout']
                train_graphs = dict(graphs, A=graphs['A'] * keep)
            score, reconstruction, _, _, covariance, alignment = forward(model, train_graphs)
            total, loss_parts = radvl_base.radvl_objective(
                model, reconstruction, positive_pairs, negative_pairs, covariance, alignment, args)
            if not torch.isfinite(total).item():
                raise FloatingPointError('Nonfinite manuscript objective; inspect posterior and loss scales')
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
            sync_device()
            recorder.epoch(training.fold, epoch, total, loss_parts, model, optimizer,
                           time.perf_counter() - epoch_started)
            sync_device(); validation_started = time.perf_counter()
            if ema is not None:
                ema.swap(model)
            with preserved_rng():
                validation_matrix = radvl_base.predict(model, graphs).detach().cpu().numpy()
            validation_scores = validation_matrix[validation.pairs[:, 0], validation.pairs[:, 1]]
            validation_metrics = ranking_metrics(validation.labels, validation_scores, threshold=None)
            decision = stopping.update(epoch, validation_metrics['AUC'])
            if decision['is_best']:
                best_state = dict(model=cpu_snapshot(model.state_dict()),
                                  optimizer=cpu_snapshot(optimizer.state_dict()), rng=capture_rng())
                best_validation_aupr = validation_metrics['AUPR']
                best_scores = validation_scores
            if ema is not None:
                ema.swap(model)
            sync_device()
            elapsed = time.perf_counter() - validation_started
            validation_seconds += elapsed
            append_csv(recorder.directory / 'validation_history.csv', [dict(
                fold=training.fold, epoch=epoch, validation_AUC=validation_metrics['AUC'],
                validation_AUPR=validation_metrics['AUPR'], best_AUC=stopping.best_auc,
                best_epoch=stopping.best_epoch, bad_epochs=stopping.bad_epochs,
                is_raw_best=decision['is_best'], patience_reset=decision['patience_reset'],
                validation_seconds=elapsed)])
            progress.update(epoch, float(total.detach().item()), validation_metrics['AUC'],
                            validation_metrics['AUPR'], stopping.best_auc, stopping.bad_epochs, args.patience)
            stopped_epoch = epoch
            if decision['should_stop']:
                break
    finally:
        progress.close()
    if best_state is None:
        raise RuntimeError('No valid validation checkpoint was selected')
    # With EMA the stored weights are the averaged ones; the optimizer state belongs to the live
    # weights and is only restored for recorder compatibility.
    model.load_state_dict(best_state['model'], strict=True)
    optimizer.load_state_dict(best_state['optimizer'])
    restore_rng(best_state['rng'])
    model.eval()
    sync_device()
    threshold, validation_f1 = validation_threshold(validation.labels, best_scores)
    append_csv(recorder.directory / 'validation_threshold.csv', [dict(
        fold=training.fold, threshold=threshold, validation_F1_at_threshold=validation_f1,
        best_epoch=stopping.best_epoch)])
    summary = dict(fold=training.fold, training_seed=args.seed, best_epoch=stopping.best_epoch,
                   stopped_epoch=stopped_epoch, max_epochs=args.epoch,
                   best_validation_AUC=stopping.best_auc,
                   validation_AUPR_at_best_AUC=best_validation_aupr,
                   patience=args.patience, min_delta=args.min_delta,
                   stop_reason='patience_exhausted' if stopping.bad_epochs >= args.patience else 'max_epochs',
                   validation_seconds_total=validation_seconds,
                   restored_model_sha256=initial_state_hash(model))
    recorder.register_selection(summary)
    print(f"Fold {training.fold+1}/{args.validation}: stopped at epoch {stopped_epoch}, "
          f"restored epoch {stopping.best_epoch}, validation AUC={stopping.best_auc:.6f}, "
          f"validation F1 threshold={threshold:.4f}", flush=True)
    return time.perf_counter() - started


_WEIGHTED.fit = fit


def write_validation_threshold_metrics(directory):
    import pandas as pd
    thresholds = pd.read_csv(directory / 'validation_threshold.csv').set_index('fold')['threshold']
    rows = []
    for fold, threshold in thresholds.items():
        labels = np.load(directory / f'true_{fold}.npy')
        scores = np.load(directory / f'pred_{fold}.npy')
        rows.append(dict(fold=fold, threshold=threshold, **ranking_metrics(labels, scores, threshold=threshold)))
    append_csv(directory / 'fold_metrics_validation_threshold.csv', rows)
    frame = pd.DataFrame(rows)
    print('TEST MEAN (validation-chosen threshold): '
          + ' | '.join(f'{name}={frame[name].mean():.6f}' for name in METRICS), flush=True)


def split_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--negative_sampling', choices=['fixed', 'epoch', 'hard'], default='fixed')
    parser.add_argument('--negative_ratio', type=float, default=1.0)
    parser.add_argument('--hard_fraction', type=float, default=0.3)
    parser.add_argument('--hard_alpha', type=float, default=0.5)
    parser.add_argument('--logit_adjust', choices=['on', 'off'], default='off')
    parser.add_argument('--edge_dropout', type=float, default=0.0)
    parser.add_argument('--ema_decay', type=float, default=0.0)
    parser.add_argument('--profile_smoothing', type=float, default=0.0)
    parser.add_argument('--profile_neighbors', type=int, default=8)
    own, rest = parser.parse_known_args(argv)
    if not 0 <= own.profile_smoothing <= 1 or own.profile_neighbors < 1:
        parser.error('Require 0 <= profile_smoothing <= 1 and profile_neighbors >= 1')
    if not own.negative_ratio > 0 or not 0 <= own.hard_fraction <= 1 or own.hard_alpha < 0:
        parser.error('Require negative_ratio > 0, 0 <= hard_fraction <= 1, hard_alpha >= 0')
    if not 0 <= own.edge_dropout < 1 or not 0 <= own.ema_decay < 1:
        parser.error('Require 0 <= edge_dropout < 1 and 0 <= ema_decay < 1')
    return own, rest


def main(args, own):
    _SAMPLER.mode, _SAMPLER.ratio = own.negative_sampling, own.negative_ratio
    _SAMPLER.hard_fraction, _SAMPLER.hard_alpha = own.hard_fraction, own.hard_alpha
    _SAMPLER.logit_adjust = own.logit_adjust == 'on'
    SETTINGS.update(edge_dropout=own.edge_dropout, ema_decay=own.ema_decay,
                    profile_smoothing=own.profile_smoothing, profile_neighbors=own.profile_neighbors)
    for name, value in vars(own).items():
        setattr(args, name, value)
    print(f'v2 controls: {vars(own)}', flush=True)
    result = controls.main(args)
    write_validation_threshold_metrics(STATE['directory'])
    return result


if __name__ == '__main__':
    own_args, remaining = split_arguments(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    main(controls.parameter_parser(), own_args)

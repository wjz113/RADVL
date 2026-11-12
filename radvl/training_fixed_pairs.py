"""RADVL training fixed pairs."""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from radvl import training_calibration as v2

controls = v2.controls
_WEIGHTED = v2._WEIGHTED
_WEIGHTED.RESULTS_ROOT = 'Results_radvl_aligned_residual_fixed11'


class FixedPairObjective:
    """Balanced per-epoch subsampling of the frozen pairs plus optional label smoothing."""

    def __init__(self, base_objective):
        self.base_objective = base_objective
        self.fraction = 1.0
        self.smoothing = 0.0
        self.generator = None

    def bind(self, training, seed):
        self.generator = np.random.default_rng([int(seed), int(training.fold), 15485863])

    def __call__(self, model, reconstruction_logits, positive_pairs, negative_pairs, covariance,
                 alignment, args):
        if model.training and self.fraction < 1:
            count = max(1, int(round(self.fraction * min(len(positive_pairs), len(negative_pairs)))))
            pick_p = torch.as_tensor(self.generator.choice(len(positive_pairs), count, replace=False),
                                     device=positive_pairs.device)
            pick_n = torch.as_tensor(self.generator.choice(len(negative_pairs), count, replace=False),
                                     device=negative_pairs.device)
            positive_pairs, negative_pairs = positive_pairs[pick_p], negative_pairs[pick_n]
        total, parts = self.base_objective(model, reconstruction_logits, positive_pairs, negative_pairs,
                                           covariance, alignment, args)
        if self.smoothing > 0:
            # Replace the hard-target prediction BCE by the smoothed one (same normalisation).
            pos = model.prediction_logits[positive_pairs[:, 0], positive_pairs[:, 1]]
            neg = model.prediction_logits[negative_pairs[:, 0], negative_pairs[:, 1]]
            e = self.smoothing
            smoothed = (F.binary_cross_entropy_with_logits(pos, torch.full_like(pos, 1 - e), reduction='sum')
                        + F.binary_cross_entropy_with_logits(neg, torch.full_like(neg, e), reduction='sum')
                        ) / (len(pos) + len(neg))
            total = total - parts['prediction_bce'] + smoothed
            parts = dict(parts, prediction_bce=smoothed)
        return total, parts


# v2's sampler stays in the chain but in 'fixed' mode it passes the frozen negatives through.
_PAIRS = FixedPairObjective(v2._SAMPLER)
controls._CONTROLS_OBJECTIVE = _PAIRS
_V2_FIT = v2.fit
_AUC_STOPPING = v2.AucEarlyStopping


class LastEpochSelection(_AUC_STOPPING):
    """``--fixed_epochs N``: train exactly N epochs and keep the last checkpoint.

    Validation AUC is still computed and logged, but it no longer selects the checkpoint, so a
    very small validation split (e.g. 1%) cannot add selection noise.  N must be chosen on a
    separate run's validation curves, never on test metrics.
    """

    def update(self, epoch, auc):
        decision = super().update(epoch, auc)
        self.best_auc, self.best_epoch = float(auc), epoch
        return dict(decision, is_best=True, should_stop=False)


def _fit(model, optimizer, training, validation, graphs, args, recorder):
    _PAIRS.bind(training, args.seed)
    return _V2_FIT(model, optimizer, training, validation, graphs, args, recorder)


_WEIGHTED.fit = _fit


def split_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pair_subsample', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--edge_dropout', type=float, default=0.0)
    parser.add_argument('--ema_decay', type=float, default=0.0)
    parser.add_argument('--fixed_epochs', type=int, default=0, help='0 = validation-AUC early stopping')
    own, rest = parser.parse_known_args(argv)
    if not 0 < own.pair_subsample <= 1 or not 0 <= own.label_smoothing < 0.5:
        parser.error('Require 0 < pair_subsample <= 1 and 0 <= label_smoothing < 0.5')
    if not 0 <= own.edge_dropout < 1 or not 0 <= own.ema_decay < 1:
        parser.error('Require 0 <= edge_dropout < 1 and 0 <= ema_decay < 1')
    return own, rest


def main(args, own):
    v2._SAMPLER.mode, v2._SAMPLER.ratio, v2._SAMPLER.logit_adjust = 'fixed', 1.0, False
    _PAIRS.fraction, _PAIRS.smoothing = own.pair_subsample, own.label_smoothing
    v2.SETTINGS.update(edge_dropout=own.edge_dropout, ema_decay=own.ema_decay, profile_smoothing=0.0)
    args.negative_sampling, args.negative_ratio, args.logit_adjust = 'fixed', 1.0, 'off'
    if own.fixed_epochs > 0:
        v2.AucEarlyStopping = LastEpochSelection
        args.epoch, args.patience = own.fixed_epochs, own.fixed_epochs + 1
    for name, value in vars(own).items():
        setattr(args, name, value)
    print(f'fixed 1:1 entry: {vars(own)} (frozen training pairs only)', flush=True)
    result = controls.main(args)
    if v2.STATE['directory'] is not None:
        v2.write_validation_threshold_metrics(v2.STATE['directory'])
    return result


if __name__ == '__main__':
    own_args, remaining = split_arguments(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    main(controls.parameter_parser(), own_args)

"""RADVL training negative sampling."""
import argparse
import sys

import numpy as np
import torch

from radvl import training_variants as controls


class ResampledNegativeObjective:
    """Replaces the frozen negative pairs with a fresh draw from the bound pool on every call."""

    def __init__(self, base_objective):
        self.base_objective = base_objective
        self.mode = 'fixed'
        self.ratio = 1.0
        self.pool = None
        self.generator = None

    def bind(self, training, seed):
        rows, columns = np.nonzero(training.allowed_negative_mask)
        self.pool = np.stack([rows, columns], axis=1)
        self.generator = np.random.default_rng([int(seed), int(training.fold), 7919])

    def __call__(self, model, reconstruction_logits, positive_pairs, negative_pairs, covariance,
                 alignment, args):
        if self.mode == 'epoch':
            count = min(len(self.pool), int(round(self.ratio * len(positive_pairs))))
            chosen = self.generator.choice(len(self.pool), size=count, replace=False)
            negative_pairs = torch.as_tensor(self.pool[chosen], dtype=torch.long,
                                             device=positive_pairs.device)
        return self.base_objective(model, reconstruction_logits, positive_pairs, negative_pairs,
                                   covariance, alignment, args)


_WEIGHTED = controls._WEIGHTED
_WEIGHTED.RESULTS_ROOT = 'Results_radvl_aligned_residual_negsample'
_SAMPLER = ResampledNegativeObjective(controls._CONTROLS_OBJECTIVE)
controls._CONTROLS_OBJECTIVE = _SAMPLER
_ORIGINAL_FIT = _WEIGHTED.fit


def _fit_with_pool(model, optimizer, training, validation, graphs, args, recorder):
    _SAMPLER.bind(training, args.seed)
    return _ORIGINAL_FIT(model, optimizer, training, validation, graphs, args, recorder)


_WEIGHTED.fit = _fit_with_pool


def split_arguments(argv):
    """Parse this entry's switches and return the rest for the controls parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--negative_sampling', choices=['fixed', 'epoch'], default='fixed')
    parser.add_argument('--negative_ratio', type=float, default=1.0)
    own, rest = parser.parse_known_args(argv)
    if not own.negative_ratio > 0:
        parser.error('negative_ratio must be positive')
    return own, rest


def main(args, own):
    _SAMPLER.mode, _SAMPLER.ratio = own.negative_sampling, own.negative_ratio
    args.negative_sampling, args.negative_ratio = own.negative_sampling, own.negative_ratio
    print(f'Negative sampling: {own.negative_sampling} (ratio {own.negative_ratio} x |P_train|, '
          f'pool = training-only allowed_negative_mask).', flush=True)
    return controls.main(args)


if __name__ == '__main__':
    own_args, remaining = split_arguments(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    main(controls.parameter_parser(), own_args)

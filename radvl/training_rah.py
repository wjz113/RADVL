"""RADVL training rah."""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn

from radvl import training_fixed_pairs as fixed11

controls = fixed11.controls
_WEIGHTED = fixed11._WEIGHTED
_WEIGHTED.RESULTS_ROOT = 'Results_radvl_aligned_residual_fixed11_hyper'
_BASE_MODEL = _WEIGHTED.RADVL
SETTINGS = dict(hap='on', hap_init=0.5, hap_neighbors=8, hap_folds=5,
                hap_assoc_weight=1.0, hap_knn_weight=1.0, hap_scales=1)


def hgnn_operator(H, weights):
    """Dv^-1/2 H W De^-1 H^T Dv^-1/2 with empty nodes/hyperedges guarded."""
    node_degree = np.maximum(H @ weights, 1e-12)
    edge_degree = np.maximum(H.sum(0), 1)
    left = H / np.sqrt(node_degree)[:, None]
    return (left * (weights / edge_degree)[None, :]) @ left.T


def knn_incidence(similarity, k):
    """Column j = hyperedge {j} plus the k most similar entities of j (self excluded from ranking)."""
    S = np.array(similarity, dtype=np.float64, copy=True)
    np.fill_diagonal(S, -np.inf)
    index = np.argsort(-S, axis=1)[:, :k]
    H = np.zeros_like(S)
    H[np.arange(len(S))[:, None], index] = 1
    np.fill_diagonal(H, 1)
    return H.T


def hap_score(A, circ_knn, drug_knn):
    """One prior (``--hap_scales 1``) or a stack of priors (``--hap_scales 3``).

    Scale stack: [joint hypergraph, association-only hypergraph, similarity-only hypergraph],
    each ``(P_c A + A P_d) / 2``; the model learns one weight per prior.
    """
    w_a, w_k = SETTINGS['hap_assoc_weight'], SETTINGS['hap_knn_weight']
    Hc, Hd = np.hstack([A, circ_knn]), np.hstack([A.T, drug_knn])
    Pc = hgnn_operator(Hc, np.r_[np.full(A.shape[1], w_a), np.full(circ_knn.shape[1], w_k)])
    Pd = hgnn_operator(Hd, np.r_[np.full(A.shape[0], w_a), np.full(drug_knn.shape[1], w_k)])
    joint = (Pc @ A + A @ Pd) / 2
    if SETTINGS['hap_scales'] == 1:
        return joint[None]
    assoc = (hgnn_operator(A, np.ones(A.shape[1])) @ A + A @ hgnn_operator(A.T, np.ones(A.shape[0]))) / 2
    knn = (hgnn_operator(circ_knn, np.ones(circ_knn.shape[1])) @ A
           + A @ hgnn_operator(drug_knn, np.ones(drug_knn.shape[1]))) / 2
    return np.stack([joint, assoc, knn])


def standardise(S):
    """Per-prior z-score over all entries; S has shape (priors, n_circ, n_drug)."""
    mean = S.mean(axis=(1, 2), keepdims=True)
    std = np.maximum(S.std(axis=(1, 2), keepdims=True), 1e-12)
    return (S - mean) / std


def hap_matrices(training, circ_similarity, drug_similarity, seed):
    """(train-time out-of-fold matrix, evaluation matrix on the full A_train)."""
    A = np.asarray(training.A_train, dtype=np.float64)
    k = SETTINGS['hap_neighbors']
    circ_knn, drug_knn = knn_incidence(circ_similarity, k), knn_incidence(drug_similarity, k)
    evaluation = standardise(hap_score(A, circ_knn, drug_knn))
    rng = np.random.default_rng([int(seed), int(training.fold), 2750159])
    positives, negatives = np.asarray(training.positive_pairs), np.asarray(training.negative_pairs)
    parts_p = rng.integers(0, SETTINGS['hap_folds'], len(positives))
    parts_n = rng.integers(0, SETTINGS['hap_folds'], len(negatives))
    train = evaluation.copy()
    for part in range(SETTINGS['hap_folds']):
        held = positives[parts_p == part]
        reduced = A.copy()
        reduced[held[:, 0], held[:, 1]] = 0
        # The similarity hyperedges stay fixed: they come from the fold's features, which the
        # released pipeline also computes once from the full A_train.
        S = standardise(hap_score(reduced, circ_knn, drug_knn))
        for pairs in (held, negatives[parts_n == part]):
            train[:, pairs[:, 0], pairs[:, 1]] = S[:, pairs[:, 0], pairs[:, 1]]
    return train, evaluation


class RADVL(_BASE_MODEL):
    """Released model plus ``beta_hap * z(S_HAP)`` on the prediction logit."""

    def __init__(self, n_circ, n_drug, args):
        super().__init__(n_circ, n_drug, args)
        self.use_hap = SETTINGS['hap'] == 'on'
        self.hap_train = self.hap_eval = None
        if self.use_hap:
            # One weight per prior; the total initial weight is hap_init, split evenly.
            scales = SETTINGS['hap_scales']
            self.hap_beta = nn.Parameter(torch.full((scales,), float(SETTINGS['hap_init']) / scales))

    def pair_logits(self, circ, drug):
        logits = super().pair_logits(circ, drug)
        if self.use_hap:
            if self.hap_eval is None:
                raise RuntimeError('HAP matrices were not bound before the forward pass')
            prior = self.hap_train if self.training else self.hap_eval
            logits = logits + torch.einsum('k,kij->ij', self.hap_beta, prior)
            with torch.no_grad():
                self.hap_logits = torch.einsum('k,kij->ij', self.hap_beta, self.hap_eval).detach()
        return logits

    def forward(self, *forward_args, **forward_kwargs):
        result = super().forward(*forward_args, **forward_kwargs)
        if self.use_hap:
            with torch.no_grad():
                self.diagnostic_scores['hap_only'] = torch.sigmoid(self.hap_logits).detach()
        return result


_WEIGHTED.RADVL = RADVL
_FIXED11_FIT = _WEIGHTED.fit


def _fit(model, optimizer, training, validation, graphs, args, recorder):
    if model.use_hap:
        train, evaluation = hap_matrices(training, graphs['circ_features'].cpu().numpy(),
                                         graphs['drug_features'].cpu().numpy(), args.seed)
        device = graphs['A'].device
        model.hap_train = torch.tensor(train, dtype=torch.float32, device=device)
        model.hap_eval = torch.tensor(evaluation, dtype=torch.float32, device=device)
    result = _FIXED11_FIT(model, optimizer, training, validation, graphs, args, recorder)
    if model.use_hap:
        print(f'Fold {training.fold + 1}: beta_hap={np.round(model.hap_beta.detach().cpu().numpy(), 4).tolist()}',
              flush=True)
    return result


_WEIGHTED.fit = _fit


def split_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--hap', choices=['on', 'off'], default='on')
    parser.add_argument('--hap_init', type=float, default=0.5)
    parser.add_argument('--hap_neighbors', type=int, default=8)
    parser.add_argument('--hap_folds', type=int, default=5)
    parser.add_argument('--hap_assoc_weight', type=float, default=1.0)
    parser.add_argument('--hap_knn_weight', type=float, default=1.0)
    parser.add_argument('--hap_scales', type=int, choices=[1, 3], default=1)
    own, rest = parser.parse_known_args(argv)
    if own.hap_neighbors < 1 or own.hap_folds < 2 or own.hap_assoc_weight < 0 or own.hap_knn_weight < 0:
        parser.error('Require hap_neighbors >= 1, hap_folds >= 2 and non-negative hyperedge weights')
    return own, rest


if __name__ == '__main__':
    hap_args, remaining = split_arguments(sys.argv[1:])
    SETTINGS.update(vars(hap_args))
    print(f'HAP switches: {SETTINGS}', flush=True)
    fixed_args, remaining = fixed11.split_arguments(remaining)
    sys.argv = [sys.argv[0]] + remaining
    parsed = controls.parameter_parser()
    for name, value in vars(hap_args).items():
        setattr(parsed, name, value)
    fixed11.main(parsed, fixed_args)

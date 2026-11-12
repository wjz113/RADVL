"""RADVL training relation gating."""
import numpy as np
import torch
import torch.nn as nn

from radvl import training_relations as relation_controls

masking_controls = relation_controls.masking_controls
controls = relation_controls.controls
_WEIGHTED = relation_controls._WEIGHTED
_RELATION_MODEL = _WEIGHTED.RADVL
import sys  # noqa: E402
hyper = sys.modules['radvl.training_rah']
DEFAULTS = dict(relgate='off', relgate_scale=1.0, relgate_input='degree')
SETTINGS = dict(DEFAULTS)


def configure(values):
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown DFX9 settings: {sorted(unknown)}')
    merged = dict(DEFAULTS, **values)
    if merged['relgate'] not in ('off', 'linear', 'sigmoid'):
        raise ValueError('relgate must be off, linear or sigmoid')
    if merged['relgate_input'] not in ('degree', 'shuffled', 'none'):
        raise ValueError('relgate_input must be degree, shuffled or none')
    merged['relgate_scale'] = float(merged['relgate_scale'])
    if not merged['relgate_scale'] > 0:
        raise ValueError('relgate_scale must be > 0')
    SETTINGS.clear()
    SETTINGS.update(merged)


def channel_scores(A, circ_knn, drug_knn):
    ones = np.ones
    pa_c = hyper.hgnn_operator(A, ones(A.shape[1]))
    pa_d = hyper.hgnn_operator(A.T, ones(A.shape[0]))
    pk_c = hyper.hgnn_operator(circ_knn, ones(circ_knn.shape[1]))
    pk_d = hyper.hgnn_operator(drug_knn, ones(drug_knn.shape[1]))
    return np.stack([pa_c @ A, A @ pa_d, pk_c @ A, A @ pk_d])


def channel_matrices(training, circ_similarity, drug_similarity, seed):
    """(train out-of-fold channels, evaluation channels); partition identical to hap_matrices."""
    A = np.asarray(training.A_train, dtype=np.float64)
    k = hyper.SETTINGS['hap_neighbors']
    circ_knn, drug_knn = hyper.knn_incidence(circ_similarity, k), hyper.knn_incidence(drug_similarity, k)
    evaluation = hyper.standardise(channel_scores(A, circ_knn, drug_knn))
    rng = np.random.default_rng([int(seed), int(training.fold), 2750159])
    positives, negatives = np.asarray(training.positive_pairs), np.asarray(training.negative_pairs)
    parts_p = rng.integers(0, hyper.SETTINGS['hap_folds'], len(positives))
    parts_n = rng.integers(0, hyper.SETTINGS['hap_folds'], len(negatives))
    train = evaluation.copy()
    for part in range(hyper.SETTINGS['hap_folds']):
        held = positives[parts_p == part]
        reduced = A.copy()
        reduced[held[:, 0], held[:, 1]] = 0
        S = hyper.standardise(channel_scores(reduced, circ_knn, drug_knn))
        for pairs in (held, negatives[parts_n == part]):
            train[:, pairs[:, 0], pairs[:, 1]] = S[:, pairs[:, 0], pairs[:, 1]]
    return train, evaluation


def _derangement(n, salt):
    order = np.random.default_rng([12289, n, salt]).permutation(n)
    perm = np.empty(n, dtype=np.int64)
    perm[order] = np.roll(order, -1)
    return perm


def reliability_matrices(training):
    """(train, eval) arrays of shape (2, n_c, n_d): r_i and r_j per pair."""
    A = np.asarray(training.A_train, dtype=np.float64)
    deg_c, deg_d = A.sum(1), A.sum(0)
    lc, ld = np.log1p(deg_c), np.log1p(deg_d)
    mc, sc, md, sd = lc.mean(), max(lc.std(), 1e-12), ld.mean(), max(ld.std(), 1e-12)
    mode = SETTINGS['relgate_input']
    if mode == 'none':
        zero = np.zeros((2,) + A.shape)
        return zero, zero.copy()
    # evaluation: the full training degree of each end node
    rc_eval = np.broadcast_to(((lc - mc) / sc)[:, None], A.shape)
    rd_eval = np.broadcast_to(((ld - md) / sd)[None, :], A.shape)
    # training: a training positive's own edge is not counted in its end nodes' degree (A is 1
    # exactly on the training positives), matching the test situation
    rc_train = (np.log1p(deg_c[:, None] - A) - mc) / sc
    rd_train = (np.log1p(deg_d[None, :] - A) - md) / sd
    if mode == 'shuffled':
        # control: every node receives the reliability of a different node (fixed derangement)
        pc, pd_ = _derangement(A.shape[0], 1), _derangement(A.shape[1], 2)
        rc_eval, rc_train = rc_eval[pc, :], rc_train[pc, :]
        rd_eval, rd_train = rd_eval[:, pd_], rd_train[:, pd_]
    return np.stack([rc_train, rd_train]), np.stack([rc_eval, rd_eval])


class RADVL(_RELATION_MODEL):
    def __init__(self, n_circ, n_drug, args):
        super().__init__(n_circ, n_drug, args)
        self.relgate = SETTINGS['relgate']
        self.relgate_scale = SETTINGS['relgate_scale']
        self.rg_train = self.rg_eval = self.rr_train = self.rr_eval = None
        if self.relgate == 'off':
            return
        if not getattr(self, 'use_hap', False):
            raise ValueError('relgate requires --hap on')
        if self.relgate == 'linear':
            self.relgate_u = nn.Parameter(torch.zeros(4))
            self.relgate_a = nn.Parameter(torch.zeros(4))
            self.relgate_b = nn.Parameter(torch.zeros(4))
        else:
            self.relgate_beta = nn.Parameter(torch.zeros(2))
            self.relgate_a = nn.Parameter(torch.zeros(2))
            self.relgate_b = nn.Parameter(torch.zeros(2))
            self.relgate_c = nn.Parameter(torch.zeros(2))

    def relgate_logits(self):
        channels = self.rg_train if self.training else self.rg_eval
        r = self.rr_train if self.training else self.rr_eval
        s = self.relgate_scale
        if self.relgate == 'linear':
            w = s * (self.relgate_u[:, None, None] + self.relgate_a[:, None, None] * r[0] + self.relgate_b[:, None, None] * r[1])
            return (w * channels).sum(0), None
        gates = []
        out = 0.0
        for side, (assoc, knn) in enumerate(((0, 2), (1, 3))):
            g = torch.sigmoid(s * (self.relgate_a[side] * r[0] + self.relgate_b[side] * r[1] + self.relgate_c[side]))
            out = out + s * self.relgate_beta[side] * (g * channels[assoc] + (1 - g) * channels[knn])
            gates.append(g)
        return out, gates

    def pair_logits(self, circ, drug):
        logits = super().pair_logits(circ, drug)
        if self.relgate == 'off':
            return logits
        if self.rg_eval is None:
            raise RuntimeError('relgate matrices were not bound before the forward pass')
        extra, gates = self.relgate_logits()
        with torch.no_grad():
            self.relgate_extra_std = float(extra.std())
            self.relgate_gates = None if gates is None else [g.detach() for g in gates]
        return logits + extra


_PREVIOUS_FIT = _WEIGHTED.fit


def _fit(model, optimizer, training, validation, graphs, args, recorder):
    if model.relgate != 'off':
        device = graphs['A'].device
        train, evaluation = channel_matrices(training, graphs['circ_features'].cpu().numpy(),
                                             graphs['drug_features'].cpu().numpy(), args.seed)
        rr_train, rr_eval = reliability_matrices(training)
        as_tensor = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device)
        model.rg_train, model.rg_eval = as_tensor(train), as_tensor(evaluation)
        model.rr_train, model.rr_eval = as_tensor(rr_train), as_tensor(rr_eval)
    result = _PREVIOUS_FIT(model, optimizer, training, validation, graphs, args, recorder)
    if model.relgate != 'off':
        from radvl.recorder_base import append_csv
        row = dict(fold=training.fold, relgate=model.relgate, scale=model.relgate_scale, input=SETTINGS['relgate_input'],
                   extra_logit_std_eval=None)
        model.eval()
        with torch.no_grad():
            extra, gates = model.relgate_logits()
            row['extra_logit_std_eval'] = float(extra.std())
            for name, p in model.named_parameters():
                if name.startswith('relgate_'):
                    for k, v in enumerate(p.detach().cpu().numpy().tolist()):
                        row[f'{name}_{k}'] = v * model.relgate_scale
            if gates is not None:
                for side, g in enumerate(gates):
                    row[f'gate_side{side}_mean'] = float(g.mean())
                    row[f'gate_side{side}_std'] = float(g.std())
        append_csv(recorder.directory / 'relgate_parameters.csv', [row])
    return result


_WEIGHTED.fit = _fit
_WEIGHTED.RADVL = RADVL
_DFX8_PARSER = controls.parameter_parser


def _parser_with_relation_gating():
    parsed = _DFX8_PARSER()
    for key, value in SETTINGS.items():
        setattr(parsed, key, value)
    return parsed


controls.parameter_parser = _parser_with_relation_gating

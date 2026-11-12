"""RADVL training variance."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from radvl import training_discrepancy as discrepancy_controls

controls = discrepancy_controls.controls
_WEIGHTED = discrepancy_controls._WEIGHTED
_DISCREPANCY_MODEL = _WEIGHTED.RADVL
DEFAULTS = dict(fusion_aux=0.0, readout_weighting='off', readout_tau_scale=1.0,
                lambda_var=0.0, var_target=0.2, view_corruption=0.0, corruption_seed=1234)
SETTINGS = dict(DEFAULTS)
FLOATS = ('fusion_aux', 'readout_tau_scale', 'lambda_var', 'var_target', 'view_corruption')


def configure(values):
    """Validate and install DFX2 settings; must be called before the model is built."""
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown DFX2 settings: {sorted(unknown)}')
    merged = dict(DEFAULTS, **values)
    for key in FLOATS:
        merged[key] = float(merged[key])
        if not math.isfinite(merged[key]):
            raise ValueError(f'{key} must be finite')
    if merged['readout_weighting'] not in ('off', 'omega', 'shuffled'):
        raise ValueError("readout_weighting must be 'off', 'omega' or 'shuffled'")
    if merged['fusion_aux'] < 0 or merged['lambda_var'] < 0:
        raise ValueError('fusion_aux and lambda_var must be >= 0')
    if merged['var_target'] <= 0 or merged['readout_tau_scale'] <= 0:
        raise ValueError('var_target and readout_tau_scale must be > 0')
    if merged['readout_weighting'] == 'off' and merged['readout_tau_scale'] != 1.0:
        raise ValueError('readout_tau_scale requires --readout_weighting omega/shuffled')
    if not 0 <= merged['view_corruption'] < 0.5:
        raise ValueError('view_corruption must be in [0, 0.5)')
    merged['corruption_seed'] = int(merged['corruption_seed'])
    SETTINGS.clear()
    SETTINGS.update(merged)


# ---------------------------------------------------------------- discrepancy-weighted readout
def node_discrepancy(fusion):
    """D_i = ||Delta_1,i||^2 + ||Delta_2,i||^2 of the current forward (Eq. 41), detached."""
    arrays = fusion.diagnostic_arrays
    return arrays['delta_1'].square().sum(dim=-1) + arrays['delta_2'].square().sum(dim=-1)


def node_weight(discrepancy, scale):
    """omega_i = exp(-D_i / (s * mean D)), detached."""
    d = discrepancy.detach()
    return torch.exp(-d / (scale * d.mean()).clamp_min(1e-12))


def derangement(n, rng):
    """A single n-cycle: a permutation without fixed points (numpy RNG, torch RNG untouched)."""
    order = rng.permutation(n)
    perm = np.arange(n)
    perm[order] = np.roll(order, -1)
    return perm


def readout_weights(circ_weight, drug_weight, perms=None):
    """w_ij = omega_i^c omega_j^d / mean(omega^c omega^d); 'shuffled' permutes the nodes first."""
    if perms is not None:
        circ_weight = circ_weight[torch.as_tensor(perms[0], dtype=torch.long, device=circ_weight.device)]
        drug_weight = drug_weight[torch.as_tensor(perms[1], dtype=torch.long, device=drug_weight.device)]
    w = circ_weight[:, None] * drug_weight[None, :]
    return (w / w.mean().clamp_min(1e-12)).detach()


class RADVL(_DISCREPANCY_MODEL):
    """DFX model plus the discrepancy-weighted readout; defaults change nothing."""

    def __init__(self, n_circ, n_drug, args):
        super().__init__(n_circ, n_drug, args)
        self.dfx2_readout = SETTINGS['readout_weighting']
        self.dfx2_tau_scale = SETTINGS['readout_tau_scale']
        self.readout_weight = None
        self.dfx2_perms = None
        if self.dfx2_readout != 'off' and not hasattr(self, 'residual_scale'):
            raise ValueError('The discrepancy-weighted readout requires --prediction_head residual_mlp')
        if self.dfx2_readout == 'shuffled':
            rng = np.random.default_rng([SETTINGS['corruption_seed'], 4057])
            self.dfx2_perms = (derangement(n_circ, rng), derangement(n_drug, rng))

    def pair_logits(self, circ, drug):
        logits = super().pair_logits(circ, drug)
        if self.dfx2_readout == 'off':
            return logits
        circ_weight = node_weight(node_discrepancy(self.circ_fusion), self.dfx2_tau_scale)
        drug_weight = node_weight(node_discrepancy(self.dr_fusion), self.dfx2_tau_scale)
        w = readout_weights(circ_weight, drug_weight, self.dfx2_perms)
        self.readout_weight = w
        with torch.no_grad():
            self.circ_fusion.diagnostic_arrays['readout_node_weight'] = circ_weight.detach()
            self.dr_fusion.diagnostic_arrays['readout_node_weight'] = drug_weight.detach()
            for fusion, node in ((self.circ_fusion, circ_weight), (self.dr_fusion, drug_weight)):
                fusion.diagnostics.update(
                    readout_node_weight_cv=(node.std() / node.mean().clamp_min(1e-12)).item(),
                    readout_pair_weight_min=w.min().item(), readout_pair_weight_max=w.max().item())
        # s_AV + gamma * s_fusion + beta * HAP  ->  s_AV + gamma * w * s_fusion + beta * HAP
        return logits + torch.tanh(self.residual_scale) * (w - 1) * self.fusion_logits


# ---------------------------------------------------------------- objective
def pair_bce(logits, positive_pairs, negative_pairs):
    """Same normalisation as the Eq. (44) prediction BCE: sum over both sets / pair count."""
    pos = logits[positive_pairs[:, 0], positive_pairs[:, 1]]
    neg = logits[negative_pairs[:, 0], negative_pairs[:, 1]]
    total = (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos), reduction='sum')
             + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg), reduction='sum'))
    return total / (len(pos) + len(neg))


def variance_hinge(mean, target):
    """VICReg variance term: mean_k relu(target - sqrt(var_k + 1e-4)), var over entities."""
    std = (mean.var(dim=0, unbiased=False) + 1e-4).sqrt()
    return F.relu(target - std).mean()


class VarianceObjective:
    def __init__(self, base_objective):
        self.base_objective = base_objective

    def __call__(self, model, reconstruction_logits, positive_pairs, negative_pairs, covariance,
                 alignment, args):
        total, parts = self.base_objective(model, reconstruction_logits, positive_pairs,
                                           negative_pairs, covariance, alignment, args)
        if SETTINGS['fusion_aux'] == 0 and SETTINGS['lambda_var'] == 0:
            return total, parts
        extra = {}
        if SETTINGS['fusion_aux'] > 0:
            bce = pair_bce(model.fusion_logits, positive_pairs, negative_pairs)
            weighted = SETTINGS['fusion_aux'] * bce
            total = total + weighted
            extra.update(fusion_aux_bce=bce.detach(), weighted_fusion_aux=weighted.detach())
        if SETTINGS['lambda_var'] > 0:
            hinge = 0.5 * (variance_hinge(model.z_node_mean, SETTINGS['var_target'])
                           + variance_hinge(model.z_edge_mean, SETTINGS['var_target']))
            weighted = SETTINGS['lambda_var'] * hinge
            total = total + weighted
            extra.update(variance_hinge=hinge.detach(), weighted_variance=weighted.detach())
        parts = dict(parts, **extra, prediction_share_of_total=parts['prediction_bce'] / total)
        return total, parts


# ---------------------------------------------------------------- view-corruption stress test
def corruption_indices(n, p, rng):
    """(chosen node indices, permutation): the chosen nodes are cycled, every other node is fixed."""
    m = int(round(p * n))
    perm = np.arange(n)
    if m < 2:
        return np.empty(0, dtype=np.int64), perm
    chosen = np.sort(rng.choice(n, m, replace=False))
    order = rng.permutation(chosen)
    perm[order] = np.roll(order, -1)
    return chosen, perm


def corruption_plan(n_circ, n_drug, p, seed, fold):
    """Deterministic per (seed, fold); used by training and by the analysis script."""
    rng = np.random.default_rng([int(seed), int(fold), 7919])
    return corruption_indices(n_circ, p, rng), corruption_indices(n_drug, p, rng)


from radvl import feature_inputs as _FEATURES  # noqa: E402  (module whose globals are patched)

_ORIGINAL_COMPUTE = _FEATURES.compute_features


def compute_features(training, circ_static, drug_static):
    features = _ORIGINAL_COMPUTE(training, circ_static, drug_static)
    p = SETTINGS['view_corruption']
    if p == 0:
        return features
    (_, circ_perm), (_, drug_perm) = corruption_plan(features['IM'].shape[0], features['ID'].shape[0],
                                                     p, SETTINGS['corruption_seed'], training.fold)
    features = dict(features)
    for key, perm in (('IM', circ_perm), ('ID', drug_perm)):
        index = torch.as_tensor(perm, dtype=torch.long)
        features[key] = features[key][index][:, index].clone()
    return features


# Both call sites: the training loop (weighted main) and label_invariance_audit (protocol module),
# so the audit recomputes the same corrupted features and still passes.
_FEATURES.compute_features = compute_features
_WEIGHTED.compute_features = compute_features
_WEIGHTED.RADVL = RADVL
controls._CONTROLS_OBJECTIVE = VarianceObjective(controls._CONTROLS_OBJECTIVE)
_DFX_PARSER = controls.parameter_parser


def _parser_with_variance():
    parsed = _DFX_PARSER()
    for key, value in SETTINGS.items():
        setattr(parsed, key, value)
    return parsed


controls.parameter_parser = _parser_with_variance

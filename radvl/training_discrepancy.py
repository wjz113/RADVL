"""RADVL training discrepancy."""
import math
from types import MethodType

import torch

from radvl import training_rah as hyper

controls = hyper.controls
_WEIGHTED = hyper._WEIGHTED
_HAP_MODEL = _WEIGHTED.RADVL
DEFAULTS = dict(decorrelation='off', lambda_dec=0.0, gate_mode='delta',
                align_weighting='adaptive', align_sensitivity=0.0)
SETTINGS = dict(DEFAULTS)
CHOICES = dict(decorrelation=('off', 'corr'), gate_mode=('delta', 'absdelta', 'zero'),
               align_weighting=('adaptive', 'standardized'))


def configure(values):
    """Validate and install DFX settings; must be called before the model is built."""
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown DFX settings: {sorted(unknown)}')
    merged = dict(DEFAULTS, **values)
    for key, allowed in CHOICES.items():
        if merged[key] not in allowed:
            raise ValueError(f'{key} must be one of {allowed}, got {merged[key]!r}')
    merged['lambda_dec'] = float(merged['lambda_dec'])
    merged['align_sensitivity'] = float(merged['align_sensitivity'])
    if merged['lambda_dec'] < 0 or not math.isfinite(merged['lambda_dec']):
        raise ValueError('lambda_dec must be finite and >= 0')
    if merged['decorrelation'] == 'off' and merged['lambda_dec'] != 0:
        raise ValueError('lambda_dec > 0 requires --decorrelation corr')
    if merged['align_weighting'] == 'adaptive' and merged['align_sensitivity'] != 0:
        raise ValueError('align_sensitivity requires --align_weighting standardized')
    SETTINGS.clear()
    SETTINGS.update(merged)


def fusion_patched():
    return SETTINGS['gate_mode'] != 'delta' or SETTINGS['align_weighting'] != 'adaptive'


def gate_signal(delta, mode):
    if mode == 'delta':
        return delta
    if mode == 'absdelta':
        return delta.abs()
    return torch.zeros_like(delta)


def block_ratio(linear, dimension):
    """||W_delta||_F / mean(||W_self||_F, ||W_context||_F) of a gate Linear(3d, d)."""
    weight = linear.weight.detach()
    delta = weight[:, :dimension].norm()
    rest = 0.5 * (weight[:, dimension:2 * dimension].norm() + weight[:, 2 * dimension:].norm())
    return float(delta / rest.clamp_min(1e-12))


def dfx_fusion_forward(self, similarity, association):
    """Balanced stop-gradient fusion forward with the two DFX substitutions; all else unchanged."""
    context_1 = self.context(similarity, association)
    context_2 = self.context(association, similarity)
    delta_1 = similarity - context_1
    delta_2 = association - context_2
    gate_1 = torch.sigmoid(self.gate_1(torch.cat(
        [gate_signal(delta_1, self.dfx_gate_mode), similarity, context_1], dim=-1)))
    gate_2 = torch.sigmoid(self.gate_2(torch.cat(
        [gate_signal(delta_2, self.dfx_gate_mode), association, context_2], dim=-1)))
    first = gate_1 * similarity + (1 - gate_1) * context_1
    second = gate_2 * association + (1 - gate_2) * context_2
    discrepancy = delta_1.square().sum(dim=-1) + delta_2.square().sum(dim=-1)
    if self.dfx_align_weighting == 'standardized':
        d = discrepancy.detach()
        z = ((d - d.mean()) / d.std().clamp_min(1e-12)).clamp(-3.0, 3.0)
        weight = torch.softmax(-self.dfx_align_sensitivity * z, dim=0) * len(z)
        # Effective temperature std(D)/|S| (finite by construction: 0 means uniform weights).
        # Must stay finite: every training_history row is asserted all-finite.
        s = abs(self.dfx_align_sensitivity)
        tau = float(d.std() / s) if s > 0 else 0.0
    else:
        if self.tau_mode == 'adaptive':
            tau = (self.tau_scale * discrepancy.mean()).detach().clamp_min(1e-12)
        else:
            tau = self.tau
        weight = torch.exp(-discrepancy / tau)
    alignment = (weight.detach() * (first - second).square().sum(dim=-1)).mean()
    fused = first + second
    dimension = similarity.shape[-1]
    with torch.no_grad():
        self.diagnostics = dict(
            delta_squared_mean=discrepancy.mean().item(),
            tau_effective=float(tau),
            alignment_strength_mean=weight.mean().item(),
            alignment_strength_min=weight.min().item(),
            alignment_strength_max=weight.max().item(),
            alignment_weight_below_1e_minus8_fraction=(weight < 1e-8).float().mean().item(),
            gate_1_mean=gate_1.mean().item(), gate_2_mean=gate_2.mean().item(),
            fused_norm_mean=fused.norm(dim=1).mean().item(),
            discrepancy_std=discrepancy.std().item(),
            alignment_weight_cv=(weight.std() / weight.mean().clamp_min(1e-12)).item(),
            gate_1_delta_block_ratio=block_ratio(self.gate_1, dimension),
            gate_2_delta_block_ratio=block_ratio(self.gate_2, dimension))
        self.diagnostic_arrays = dict(
            delta_1=delta_1.detach(), delta_2=delta_2.detach(),
            gate_1=gate_1.detach(), gate_2=gate_2.detach(),
            alignment_strength=weight.detach())
    return fused, alignment


class RADVL(_HAP_MODEL):
    """HAP model; when a DFX fusion switch is on, both fusion modules get dfx_fusion_forward."""

    def __init__(self, n_circ, n_drug, args):
        super().__init__(n_circ, n_drug, args)
        if fusion_patched():
            for fusion in (self.circ_fusion, self.dr_fusion):
                fusion.dfx_gate_mode = SETTINGS['gate_mode']
                fusion.dfx_align_weighting = SETTINGS['align_weighting']
                fusion.dfx_align_sensitivity = SETTINGS['align_sensitivity']
                fusion.forward = MethodType(dfx_fusion_forward, fusion)


def correlation_penalty(mean):
    """mean_{a != b} r_ab^2 of the column-standardized matrix (rows = entities)."""
    centered = mean - mean.mean(dim=0, keepdim=True)
    standardized = centered / centered.square().mean(dim=0, keepdim=True).add(1e-8).sqrt()
    correlation = standardized.T @ standardized / len(mean)
    dimension = correlation.shape[0]
    off_diagonal = correlation - torch.diag_embed(correlation.diagonal())
    return off_diagonal.square().sum() / (dimension * (dimension - 1))


class DiscrepancyObjective:
    def __init__(self, base_objective):
        self.base_objective = base_objective

    def __call__(self, model, reconstruction_logits, positive_pairs, negative_pairs, covariance,
                 alignment, args):
        if SETTINGS['decorrelation'] == 'off':
            return self.base_objective(model, reconstruction_logits, positive_pairs, negative_pairs,
                                       covariance, alignment, args)
        total, parts = self.base_objective(model, reconstruction_logits, positive_pairs,
                                           negative_pairs, torch.zeros_like(covariance), alignment, args)
        penalty = 0.5 * (correlation_penalty(model.z_node_mean) + correlation_penalty(model.z_edge_mean))
        weighted = SETTINGS['lambda_dec'] * penalty
        total = total + weighted
        parts = dict(parts, covariance_raw=covariance.detach(), decorrelation_raw=penalty.detach(),
                     weighted_decorrelation=weighted.detach(),
                     prediction_share_of_total=parts['prediction_bce'] / total)
        return total, parts


_WEIGHTED.RADVL = RADVL
controls._CONTROLS_OBJECTIVE = DiscrepancyObjective(controls._CONTROLS_OBJECTIVE)
_ORIGINAL_PARSER = controls.parameter_parser


def _parser_with_discrepancy():
    parsed = _ORIGINAL_PARSER()
    for key, value in SETTINGS.items():
        setattr(parsed, key, value)
    return parsed


controls.parameter_parser = _parser_with_discrepancy

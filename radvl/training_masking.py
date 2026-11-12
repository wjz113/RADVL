"""RADVL training masking."""
import math
from types import MethodType

import torch

from radvl import training_context as context_controls

token_controls = context_controls.token_controls
readout_controls = token_controls.readout_controls
discrepancy_controls = token_controls.discrepancy_controls
controls = context_controls.controls
_WEIGHTED = context_controls._WEIGHTED
_CONTEXT_MODEL = _WEIGHTED.RADVL
DEFAULTS = dict(align_context_grad_scale=1.0)
SETTINGS = dict(DEFAULTS)


def configure(values):
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown DFX7 settings: {sorted(unknown)}')
    merged = dict(DEFAULTS, **values)
    merged['align_context_grad_scale'] = float(merged['align_context_grad_scale'])
    if not 0 <= merged['align_context_grad_scale'] <= 1:
        raise ValueError('align_context_grad_scale must be in [0, 1]')
    SETTINGS.clear()
    SETTINGS.update(merged)


def scaled(x, rho):
    return x.detach() + rho * (x - x.detach())


def dfx7_fusion_forward(self, similarity, association):
    if token_controls.SETTINGS['token_attention'] != 'off':
        context_1, stats_1 = token_controls.token_context(self, similarity, association, 'direction_1')
        context_2, stats_2 = token_controls.token_context(self, association, similarity, 'direction_2')
    else:
        context_1, stats_1 = self.context(similarity, association), {}
        context_2, stats_2 = self.context(association, similarity), {}
    delta_1 = similarity - context_1
    delta_2 = association - context_2
    gate_1 = torch.sigmoid(self.gate_1(torch.cat([discrepancy_controls.gate_signal(delta_1, self.dfx_gate_mode), similarity, context_1], dim=-1)))
    gate_2 = torch.sigmoid(self.gate_2(torch.cat([discrepancy_controls.gate_signal(delta_2, self.dfx_gate_mode), association, context_2], dim=-1)))
    first = gate_1 * similarity + (1 - gate_1) * context_1
    second = gate_2 * association + (1 - gate_2) * context_2
    discrepancy = delta_1.square().sum(dim=-1) + delta_2.square().sum(dim=-1)
    if self.dfx_align_weighting == 'standardized':
        d = discrepancy.detach()
        z = ((d - d.mean()) / d.std().clamp_min(1e-12)).clamp(-3.0, 3.0)
        weight = torch.softmax(-self.dfx_align_sensitivity * z, dim=0) * len(z)
        s = abs(self.dfx_align_sensitivity)
        tau = float(d.std() / s) if s > 0 else 0.0
    else:
        tau = (self.tau_scale * discrepancy.mean()).detach().clamp_min(1e-12) if self.tau_mode == 'adaptive' else self.tau
        weight = torch.exp(-discrepancy / tau)
    rho = SETTINGS['align_context_grad_scale']
    c1, c2 = scaled(context_1, rho), scaled(context_2, rho)
    gate_1a = torch.sigmoid(self.gate_1(torch.cat([discrepancy_controls.gate_signal(similarity - c1, self.dfx_gate_mode), similarity, c1], dim=-1)))
    gate_2a = torch.sigmoid(self.gate_2(torch.cat([discrepancy_controls.gate_signal(association - c2, self.dfx_gate_mode), association, c2], dim=-1)))
    first_align = gate_1a * similarity + (1 - gate_1a) * c1
    second_align = gate_2a * association + (1 - gate_2a) * c2
    alignment = (weight.detach() * (first_align - second_align).square().sum(dim=-1)).mean()
    fused = first + second
    dimension = similarity.shape[-1]
    with torch.no_grad():
        def spec(x):
            c = x - x.mean(0, keepdim=True)
            return (c.norm(dim=1).mean() / x.norm(dim=1).mean().clamp_min(1e-12)).item()
        self.diagnostics = dict(
            delta_squared_mean=discrepancy.mean().item(), tau_effective=float(tau),
            alignment_strength_mean=weight.mean().item(), alignment_strength_min=weight.min().item(),
            alignment_strength_max=weight.max().item(),
            alignment_weight_below_1e_minus8_fraction=(weight < 1e-8).float().mean().item(),
            gate_1_mean=gate_1.mean().item(), gate_2_mean=gate_2.mean().item(),
            fused_norm_mean=fused.norm(dim=1).mean().item(), discrepancy_std=discrepancy.std().item(),
            alignment_weight_cv=(weight.std() / weight.mean().clamp_min(1e-12)).item(),
            gate_1_delta_block_ratio=discrepancy_controls.block_ratio(self.gate_1, dimension),
            gate_2_delta_block_ratio=discrepancy_controls.block_ratio(self.gate_2, dimension),
            context_1_node_specificity=spec(context_1), context_2_node_specificity=spec(context_2),
            context_1_norm_mean=context_1.norm(dim=1).mean().item(), context_2_norm_mean=context_2.norm(dim=1).mean().item(),
            **{'token1_' + k: v for k, v in stats_1.items() if not math.isnan(v)},
            **{'token2_' + k: v for k, v in stats_2.items() if not math.isnan(v)})
        self.diagnostic_arrays = dict(
            delta_1=delta_1.detach(), delta_2=delta_2.detach(),
            gate_1=gate_1.detach(), gate_2=gate_2.detach(), alignment_strength=weight.detach())
    return fused, alignment


class RADVL(_CONTEXT_MODEL):
    def __init__(self, n_circ, n_drug, args):
        super().__init__(n_circ, n_drug, args)
        if SETTINGS['align_context_grad_scale'] == 1.0:
            return
        if context_controls.SETTINGS['align_context_grad'] != 'on':
            raise ValueError('use either DFX6 align_context_grad off or DFX7 align_context_grad_scale, not both')
        if readout_controls.SETTINGS['gate_dscale'] != 'off':
            raise ValueError('not combined with DFX4 gate_dscale')
        for fusion in (self.circ_fusion, self.dr_fusion):
            fusion.dfx_gate_mode = discrepancy_controls.SETTINGS['gate_mode']
            fusion.dfx_align_weighting = discrepancy_controls.SETTINGS['align_weighting']
            fusion.dfx_align_sensitivity = discrepancy_controls.SETTINGS['align_sensitivity']
            fusion.forward = MethodType(dfx7_fusion_forward, fusion)


_WEIGHTED.RADVL = RADVL
_DFX6_PARSER = controls.parameter_parser


def _parser_with_masking():
    parsed = _DFX6_PARSER()
    for key, value in SETTINGS.items():
        setattr(parsed, key, value)
    return parsed


controls.parameter_parser = _parser_with_masking

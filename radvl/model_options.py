"""RADVL model options."""
from types import MethodType

import torch
from torch import nn

from radvl import training_relation_gating as relation_gating_controls
from radvl import protocol_audit as protocol_audit

relation_controls = relation_gating_controls.relation_controls
token_controls = relation_controls.masking_controls.context_controls.token_controls
readout_controls = token_controls.readout_controls
variance_controls = readout_controls.variance_controls
discrepancy_controls = variance_controls.discrepancy_controls
controls = discrepancy_controls.controls
_WEIGHTED = discrepancy_controls._WEIGHTED
_BASE_MODEL = _WEIGHTED.RADVL
DEFAULTS = dict(abl_fusion='off', abl_view='both', abl_gin='kan', abl_hgcn='on', abl_kangin='on')
CHOICES = dict(abl_fusion=('off', 'uniform_omega', 'const_gate', 'no_cross_attention', 'additive'),
               abl_view=('both', 'no_similarity', 'no_association'),
               abl_gin=('kan', 'mlp'), abl_hgcn=('on', 'off'), abl_kangin=('on', 'off'))
SETTINGS = dict(DEFAULTS)

protocol_audit.install()


def configure(values):
    """Validate and install the review settings; must be called before the model is built."""
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown review settings: {sorted(unknown)}')
    merged = dict(DEFAULTS, **values)
    for key, allowed in CHOICES.items():
        if merged[key] not in allowed:
            raise ValueError(f'{key} must be one of {allowed}, got {merged[key]!r}')
    if merged['abl_view'] != 'both' and merged['abl_fusion'] != 'off':
        raise ValueError('abl_view already removes the fusion mechanisms; keep abl_fusion off')
    if merged['abl_kangin'] == 'off' and merged['abl_hgcn'] == 'off':
        raise ValueError('removing both KAN-GIN and HGCN is abl_view=no_similarity')
    SETTINGS.clear()
    SETTINGS.update(merged)


def fusion_replaced():
    return SETTINGS['abl_fusion'] != 'off' or SETTINGS['abl_view'] != 'both'


def review_fusion_forward(self, similarity, association):
    """CF fusion forward (stop-gradient omega, |Delta| gate) with one mechanism removed."""
    mode, view = self.review_fusion, self.review_view
    zero = torch.zeros_like(similarity)
    if view == 'both' and mode != 'additive':
        if mode == 'no_cross_attention':
            context_1, context_2 = association, similarity
        else:
            context_1 = self.context(similarity, association)
            context_2 = self.context(association, similarity)
        delta_1 = similarity - context_1
        delta_2 = association - context_2
        if mode == 'const_gate':
            gate_1 = torch.full_like(similarity, 0.5)
            gate_2 = torch.full_like(association, 0.5)
        else:
            gate_1 = torch.sigmoid(self.gate_1(torch.cat(
                [discrepancy_controls.gate_signal(delta_1, self.dfx_gate_mode), similarity, context_1], dim=-1)))
            gate_2 = torch.sigmoid(self.gate_2(torch.cat(
                [discrepancy_controls.gate_signal(delta_2, self.dfx_gate_mode), association, context_2], dim=-1)))
        first = gate_1 * similarity + (1 - gate_1) * context_1
        second = gate_2 * association + (1 - gate_2) * context_2
        discrepancy = delta_1.square().sum(dim=-1) + delta_2.square().sum(dim=-1)
        if self.tau_mode == 'adaptive':
            tau = (self.tau_scale * discrepancy.mean()).detach().clamp_min(1e-12)
        else:
            tau = self.tau
        weight = torch.exp(-discrepancy / tau)
        if mode == 'uniform_omega':
            weight = weight.detach().mean().expand_as(weight)
        alignment = (weight.detach() * (first - second).square().sum(dim=-1)).mean()
    else:
        # No second view to attend to, gate or align with: attention, gates and alignment are gone.
        context_1 = context_2 = zero
        delta_1, delta_2 = similarity, association
        ones, zeros = torch.ones_like(similarity), torch.zeros_like(similarity)
        if view == 'no_similarity':
            first, second, gate_1, gate_2 = zero, association, zeros, ones
        elif view == 'no_association':
            first, second, gate_1, gate_2 = similarity, zero, ones, zeros
        else:
            first, second, gate_1, gate_2 = similarity, association, ones, ones
        discrepancy = delta_1.square().sum(dim=-1) + delta_2.square().sum(dim=-1)
        tau = (self.tau_scale * discrepancy.mean()).detach().clamp_min(1e-12)
        weight = torch.ones_like(discrepancy)
        alignment = similarity.new_zeros(())
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
            gate_1_delta_block_ratio=discrepancy_controls.block_ratio(self.gate_1, dimension),
            gate_2_delta_block_ratio=discrepancy_controls.block_ratio(self.gate_2, dimension))
        self.diagnostic_arrays = dict(
            delta_1=delta_1.detach(), delta_2=delta_2.detach(),
            gate_1=gate_1.detach(), gate_2=gate_2.detach(),
            alignment_strength=weight.detach().clone())
    return fused, alignment


def _zero_output(module, inputs, output):
    return torch.zeros_like(output)


class RADVL(_BASE_MODEL):
    """CF model plus the review ablation switches; defaults leave the CF model untouched."""

    def __init__(self, n_circ, n_drug, args):
        super().__init__(n_circ, n_drug, args)
        if SETTINGS['abl_gin'] == 'mlp':
            for encoder in (self.circ_gin_encoder, self.dr_gin_encoder):
                for layer in encoder.layers:
                    width = layer.mlp[3].in_features
                    layer.mlp[0] = nn.Linear(width, width)
        if SETTINGS['abl_hgcn'] == 'off':
            for projection in (self.circ_global_projection, self.dr_global_projection):
                projection.register_forward_hook(_zero_output)
        if SETTINGS['abl_kangin'] == 'off':
            for encoder in (self.circ_gin_encoder, self.dr_gin_encoder):
                encoder.register_forward_hook(_zero_output)
        if SETTINGS['abl_view'] == 'no_association':
            problems = []
            if self.prediction_head != 'mlp':
                problems.append('prediction_head=mlp')
            if args.lambda_av != 0 or args.lambda_cov != 0:
                problems.append('lambda_av=0 and lambda_cov=0')
            if discrepancy_controls.SETTINGS['decorrelation'] != 'off' or variance_controls.SETTINGS['lambda_var'] != 0:
                problems.append('decorrelation=off and lambda_var=0')
            if problems:
                raise ValueError('abl_view=no_association requires ' + ', '.join(problems))
        if fusion_replaced():
            if token_controls.SETTINGS['token_attention'] != 'off' or readout_controls.SETTINGS['gate_dscale'] != 'off':
                raise ValueError('review fusion ablations are not combined with token attention or gate_dscale')
            for fusion in (self.circ_fusion, self.dr_fusion):
                fusion.review_fusion = SETTINGS['abl_fusion']
                fusion.review_view = SETTINGS['abl_view']
                fusion.dfx_gate_mode = discrepancy_controls.SETTINGS['gate_mode']
                fusion.forward = MethodType(review_fusion_forward, fusion)


_WEIGHTED.RADVL = RADVL
_PREVIOUS_PARSER = controls.parameter_parser


def _parser_with_review():
    parsed = _PREVIOUS_PARSER()
    for key, value in SETTINGS.items():
        setattr(parsed, key, value)
    return parsed


controls.parameter_parser = _parser_with_review

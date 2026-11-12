"""RADVL training token attention."""
import math
from types import MethodType

import torch
import torch.nn as nn

from radvl import training_readout as readout_controls

variance_controls = readout_controls.variance_controls
discrepancy_controls = variance_controls.discrepancy_controls
controls = readout_controls.controls
_WEIGHTED = readout_controls._WEIGHTED
_READOUT_MODEL = _WEIGHTED.RADVL
DEFAULTS = dict(token_attention='off', token_groups=8, token_norm='rms', token_lambda_init=0.2)
SETTINGS = dict(DEFAULTS)


def configure(values):
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown DFX5 settings: {sorted(unknown)}')
    merged = dict(DEFAULTS, **values)
    if merged['token_attention'] not in ('off', 'diff', 'sum', 'single'):
        raise ValueError('token_attention must be off, diff, sum or single')
    if merged['token_norm'] not in ('rms', 'rowsum'):
        raise ValueError('token_norm must be rms or rowsum')
    merged['token_groups'] = int(merged['token_groups'])
    merged['token_lambda_init'] = float(merged['token_lambda_init'])
    if merged['token_groups'] < 2 or not 0 <= merged['token_lambda_init'] < 1:
        raise ValueError('token_groups >= 2 and 0 <= token_lambda_init < 1 required')
    SETTINGS.clear()
    SETTINGS.update(merged)


class TokenDifferential(nn.Module):
    """lambda re-parameterisation and RMSNorm weight of one attention direction."""

    def __init__(self, token_dim, lambda_init, generator):
        super().__init__()
        half = token_dim // 2
        self.lambda_init = lambda_init
        self.lambda_q1 = nn.Parameter(torch.randn(half, generator=generator) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(half, generator=generator) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(half, generator=generator) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(half, generator=generator) * 0.1)
        self.norm_weight = nn.Parameter(torch.ones(token_dim))

    def value(self):
        return (torch.exp((self.lambda_q1 * self.lambda_k1).sum()) - torch.exp((self.lambda_q2 * self.lambda_k2).sum())
                + self.lambda_init)


def token_context(fusion, queries, keyside, direction):
    """Context of every node from the G feature tokens of its own other-view vector."""
    n, d = queries.shape
    g = SETTINGS['token_groups']
    t = d // g
    q = fusion.query(queries).view(n, g, t)
    k = fusion.key(keyside).view(n, g, t)
    v = fusion.value(keyside).view(n, g, t)
    mode = SETTINGS['token_attention']
    stats = {}
    if mode == 'single':
        attention = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(t), dim=-1)
        out = attention @ v
        stats.update(lambda_value=float('nan'))
    else:
        h = t // 2
        a1 = torch.softmax(q[..., :h] @ k[..., :h].transpose(1, 2) / math.sqrt(h), dim=-1)
        a2 = torch.softmax(q[..., h:] @ k[..., h:].transpose(1, 2) / math.sqrt(h), dim=-1)
        module = getattr(fusion, 'dfx5_' + direction)
        lam = module.value()
        sign = -1.0 if mode == 'diff' else 1.0
        attention = a1 + sign * lam * a2
        out = attention @ v
        if SETTINGS['token_norm'] == 'rms':
            out = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + 1e-5) * module.norm_weight
            out = out * (1.0 - module.lambda_init)
        else:
            out = out / (1.0 + sign * lam)
        with torch.no_grad():
            stats.update(lambda_value=float(lam), map_cosine=float(torch.nn.functional.cosine_similarity(
                a1.reshape(n, -1), a2.reshape(n, -1), dim=1).mean()))
    with torch.no_grad():
        positive = attention.clamp_min(1e-12)
        weights = positive / positive.sum(-1, keepdim=True)
        stats.update(entropy=float((-(weights * weights.log()).sum(-1) / math.log(g)).mean()),
                     negative_fraction=float((attention < 0).float().mean()),
                     node_row_cosine=float(_mean_pairwise_cosine(attention.reshape(n, -1))))
    return out.reshape(n, d), stats


def _mean_pairwise_cosine(x):
    x = torch.nn.functional.normalize(x - 0.0, dim=1)
    s = x @ x.T
    n = len(x)
    return (s.sum() - n) / (n * (n - 1))


def dfx5_fusion_forward(self, similarity, association):
    """dfx_fusion_forward with the context replaced by token-level (differential) attention."""
    context_1, stats_1 = token_context(self, similarity, association, 'direction_1')
    context_2, stats_2 = token_context(self, association, similarity, 'direction_2')
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
    alignment = (weight.detach() * (first - second).square().sum(dim=-1)).mean()
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
            **{'token1_' + k: v for k, v in stats_1.items() if not math.isnan(v)},
            **{'token2_' + k: v for k, v in stats_2.items() if not math.isnan(v)})
        self.diagnostic_arrays = dict(
            delta_1=delta_1.detach(), delta_2=delta_2.detach(),
            gate_1=gate_1.detach(), gate_2=gate_2.detach(), alignment_strength=weight.detach())
    return fused, alignment


class RADVL(_READOUT_MODEL):
    def __init__(self, n_circ, n_drug, args):
        super().__init__(n_circ, n_drug, args)
        mode = SETTINGS['token_attention']
        if mode == 'off':
            return
        if readout_controls.SETTINGS['gate_dscale'] != 'off':
            raise ValueError('token attention is not combined with DFX4 gate_dscale')
        generator = torch.Generator().manual_seed(4099)
        for fusion in (self.circ_fusion, self.dr_fusion):
            d = fusion.query.in_features
            g = SETTINGS['token_groups']
            if d % g or (d // g) % 2:
                raise ValueError('d must be divisible by token_groups and d / token_groups must be even')
            if mode in ('diff', 'sum'):
                fusion.dfx5_direction_1 = TokenDifferential(d // g, SETTINGS['token_lambda_init'], generator)
                fusion.dfx5_direction_2 = TokenDifferential(d // g, SETTINGS['token_lambda_init'], generator)
            fusion.dfx_gate_mode = discrepancy_controls.SETTINGS['gate_mode']
            fusion.dfx_align_weighting = discrepancy_controls.SETTINGS['align_weighting']
            fusion.dfx_align_sensitivity = discrepancy_controls.SETTINGS['align_sensitivity']
            fusion.forward = MethodType(dfx5_fusion_forward, fusion)


_WEIGHTED.RADVL = RADVL
_DFX4_PARSER = controls.parameter_parser


def _parser_with_token_attention():
    parsed = _DFX4_PARSER()
    for key, value in SETTINGS.items():
        setattr(parsed, key, value)
    return parsed


controls.parameter_parser = _parser_with_token_attention

"""RADVL training readout."""
import math
from types import MethodType

import numpy as np
import torch
import torch.nn as nn

from radvl import training_variance as variance_controls

discrepancy_controls = variance_controls.discrepancy_controls
controls = variance_controls.controls
_WEIGHTED = variance_controls._WEIGHTED
_VARIANCE_MODEL = _WEIGHTED.RADVL
DEFAULTS = dict(view_dropout=0.0, gate_dscale='off', readout_arbitrate='off', cold_probe_fraction=0.2)
SETTINGS = dict(DEFAULTS)


def configure(values):
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown DFX4 settings: {sorted(unknown)}')
    merged = dict(DEFAULTS, **values)
    merged['view_dropout'] = float(merged['view_dropout'])
    merged['cold_probe_fraction'] = float(merged['cold_probe_fraction'])
    if not 0 <= merged['view_dropout'] < 0.9 or not math.isfinite(merged['view_dropout']):
        raise ValueError('view_dropout must be in [0, 0.9)')
    if not 0 < merged['cold_probe_fraction'] < 1:
        raise ValueError('cold_probe_fraction must be in (0, 1)')
    for key in ('gate_dscale', 'readout_arbitrate'):
        if merged[key] not in ('off', 'on'):
            raise ValueError(f'{key} must be off or on')
    SETTINGS.clear()
    SETTINGS.update(merged)


def zlog(discrepancy):
    """Detached per-batch z-score of log D, clipped to +-3."""
    x = discrepancy.detach().clamp_min(1e-12).log()
    return ((x - x.mean()) / x.std().clamp_min(1e-12)).clamp(-3.0, 3.0)


def dfx4_fusion_forward(self, similarity, association):
    """dfx_fusion_forward plus the additive a_k * z(log D) gate term; all else unchanged."""
    context_1 = self.context(similarity, association)
    context_2 = self.context(association, similarity)
    delta_1 = similarity - context_1
    delta_2 = association - context_2
    discrepancy = delta_1.square().sum(dim=-1) + delta_2.square().sum(dim=-1)
    zd = zlog(discrepancy)[:, None]
    gate_1 = torch.sigmoid(self.gate_1(torch.cat(
        [discrepancy_controls.gate_signal(delta_1, self.dfx_gate_mode), similarity, context_1], dim=-1)) + zd * self.dfx4_gate_scale_1)
    gate_2 = torch.sigmoid(self.gate_2(torch.cat(
        [discrepancy_controls.gate_signal(delta_2, self.dfx_gate_mode), association, context_2], dim=-1)) + zd * self.dfx4_gate_scale_2)
    first = gate_1 * similarity + (1 - gate_1) * context_1
    second = gate_2 * association + (1 - gate_2) * context_2
    if self.dfx_align_weighting == 'standardized':
        d = discrepancy.detach()
        z = ((d - d.mean()) / d.std().clamp_min(1e-12)).clamp(-3.0, 3.0)
        weight = torch.softmax(-self.dfx_align_sensitivity * z, dim=0) * len(z)
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
            delta_squared_mean=discrepancy.mean().item(), tau_effective=float(tau),
            alignment_strength_mean=weight.mean().item(), alignment_strength_min=weight.min().item(),
            alignment_strength_max=weight.max().item(),
            alignment_weight_below_1e_minus8_fraction=(weight < 1e-8).float().mean().item(),
            gate_1_mean=gate_1.mean().item(), gate_2_mean=gate_2.mean().item(),
            fused_norm_mean=fused.norm(dim=1).mean().item(), discrepancy_std=discrepancy.std().item(),
            alignment_weight_cv=(weight.std() / weight.mean().clamp_min(1e-12)).item(),
            gate_1_delta_block_ratio=discrepancy_controls.block_ratio(self.gate_1, dimension),
            gate_2_delta_block_ratio=discrepancy_controls.block_ratio(self.gate_2, dimension),
            gate_dscale_1_norm=self.dfx4_gate_scale_1.norm().item(),
            gate_dscale_2_norm=self.dfx4_gate_scale_2.norm().item())
        self.diagnostic_arrays = dict(
            delta_1=delta_1.detach(), delta_2=delta_2.detach(),
            gate_1=gate_1.detach(), gate_2=gate_2.detach(), alignment_strength=weight.detach())
    return fused, alignment


class RADVL(_VARIANCE_MODEL):
    def __init__(self, n_circ, n_drug, args):
        super().__init__(n_circ, n_drug, args)
        self.dfx4_view_dropout = SETTINGS['view_dropout']
        self.dfx4_generator = None
        self.dfx4_arbitrate = SETTINGS['readout_arbitrate'] == 'on'
        if SETTINGS['gate_dscale'] == 'on':
            for fusion in (self.circ_fusion, self.dr_fusion):
                width = fusion.gate_1.out_features
                fusion.dfx4_gate_scale_1 = nn.Parameter(torch.zeros(width))
                fusion.dfx4_gate_scale_2 = nn.Parameter(torch.zeros(width))
                fusion.dfx_gate_mode = discrepancy_controls.SETTINGS['gate_mode']
                fusion.dfx_align_weighting = discrepancy_controls.SETTINGS['align_weighting']
                fusion.dfx_align_sensitivity = discrepancy_controls.SETTINGS['align_sensitivity']
                fusion.forward = MethodType(dfx4_fusion_forward, fusion)
        if self.dfx4_arbitrate:
            if not hasattr(self, 'residual_scale'):
                raise ValueError('readout_arbitrate requires --prediction_head residual_mlp')
            self.dfx4_kappa = nn.Parameter(torch.zeros(1))

    def forward(self, A, *rest, **kwargs):
        if self.training and self.dfx4_view_dropout > 0:
            if self.dfx4_generator is None:
                raise RuntimeError('view dropout generator was not bound')
            g = self.dfx4_generator
            keep_c = (torch.rand(A.shape[0], generator=g, device=A.device) >= self.dfx4_view_dropout).to(A.dtype)
            keep_d = (torch.rand(A.shape[1], generator=g, device=A.device) >= self.dfx4_view_dropout).to(A.dtype)
            A = A * keep_c[:, None] * keep_d[None, :]
        return super().forward(A, *rest, **kwargs)

    def pair_logits(self, circ, drug):
        logits = super().pair_logits(circ, drug)
        if not self.dfx4_arbitrate:
            return logits
        zc = zlog(variance_controls.node_discrepancy(self.circ_fusion))
        zd = zlog(variance_controls.node_discrepancy(self.dr_fusion))
        u = 0.5 * (zc[:, None] + zd[None, :])
        gamma = torch.tanh(self.residual_scale)
        return logits + self.dfx4_kappa * u * (gamma * self.fusion_logits - self.association_logits())


# ---------------------------------------------------------------- fit wrapper + cold probe
from radvl import training_core as _MANUSCRIPT  # noqa: E402
from radvl.optimization import preserved_rng  # noqa: E402
from radvl.recorder_base import append_csv  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

_PREVIOUS_FIT = _WEIGHTED.fit


def cold_masks(n_circ, n_drug, fraction, seed, fold):
    rng = np.random.default_rng([int(seed), int(fold), 60013])
    circ = np.zeros(n_circ, dtype=bool)
    circ[rng.choice(n_circ, int(round(fraction * n_circ)), replace=False)] = True
    drug = np.zeros(n_drug, dtype=bool)
    drug[rng.choice(n_drug, int(round(fraction * n_drug)), replace=False)] = True
    return circ, drug


def _gate_node_means(fusion):
    arrays = fusion.diagnostic_arrays
    return arrays['gate_1'].mean(dim=1).cpu().numpy(), arrays['gate_2'].mean(dim=1).cpu().numpy()


def cold_probe(model, graphs, validation, args, fold, directory):
    A = graphs['A']
    circ_mask, drug_mask = cold_masks(A.shape[0], A.shape[1], SETTINGS['cold_probe_fraction'], args.seed, fold)
    row = dict(fold=fold)
    saved = dict(circ_mask=circ_mask, drug_mask=drug_mask)
    pairs, labels = validation.pairs, validation.labels
    for side, mask in (('circ', circ_mask), ('drug', drug_mask)):
        masked = A.clone()
        index = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long, device=A.device)
        if side == 'circ':
            masked[index, :] = 0
        else:
            masked[:, index] = 0
        with preserved_rng(), torch.no_grad():
            scores = _MANUSCRIPT.predict(model, dict(graphs, A=masked)).detach().cpu().numpy()
        member = mask[pairs[:, 0]] if side == 'circ' else mask[pairs[:, 1]]
        row[side + '_cold_pairs'] = int(member.sum())
        row[side + '_cold_val_AUC'] = (float(roc_auc_score(labels[member], scores[pairs[member, 0], pairs[member, 1]]))
                                       if len(np.unique(labels[member])) == 2 else float('nan'))
        fusion = model.circ_fusion if side == 'circ' else model.dr_fusion
        D = variance_controls.node_discrepancy(fusion).cpu().numpy()
        omega = fusion.diagnostic_arrays['alignment_strength'].cpu().numpy()
        g1, g2 = _gate_node_means(fusion)
        row[side + '_D_AUROC_masked'] = float(roc_auc_score(mask, D))
        row[side + '_negomega_AUROC_masked'] = float(roc_auc_score(mask, -omega))
        row[side + '_gate1_masked_minus_other'] = float(g1[mask].mean() - g1[~mask].mean())
        row[side + '_gate2_masked_minus_other'] = float(g2[mask].mean() - g2[~mask].mean())
        saved.update({side + '_cold_scores': scores, side + '_cold_D': D, side + '_cold_omega': omega,
                      side + '_cold_gate1': g1, side + '_cold_gate2': g2})
    if hasattr(model, 'dfx4_kappa'):
        row['kappa'] = float(model.dfx4_kappa.detach().item())
    if hasattr(model.circ_fusion, 'dfx4_gate_scale_1'):
        for name, fusion in (('circ', model.circ_fusion), ('drug', model.dr_fusion)):
            row[name + '_gate_dscale_1_mean'] = float(fusion.dfx4_gate_scale_1.detach().mean().item())
            row[name + '_gate_dscale_2_mean'] = float(fusion.dfx4_gate_scale_2.detach().mean().item())
    np.savez_compressed(directory / f'cold_probe_{fold}.npz', **saved)
    append_csv(directory / 'cold_probe.csv', [row])


def _fit(model, optimizer, training, validation, graphs, args, recorder):
    if model.dfx4_view_dropout > 0:
        generator = torch.Generator(device=graphs['A'].device)
        generator.manual_seed(int(args.seed) * 100003 + int(training.fold) * 7 + 3)
        model.dfx4_generator = generator
    result = _PREVIOUS_FIT(model, optimizer, training, validation, graphs, args, recorder)
    was_training = model.training
    cold_probe(model, graphs, validation, args, training.fold, recorder.directory)
    model.train(was_training)
    return result


_WEIGHTED.fit = _fit
_WEIGHTED.RADVL = RADVL
_DFX2_PARSER = controls.parameter_parser


def _parser_with_readout():
    parsed = _DFX2_PARSER()
    for key, value in SETTINGS.items():
        setattr(parsed, key, value)
    return parsed


controls.parameter_parser = _parser_with_readout

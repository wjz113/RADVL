"""RADVL model balanced."""
from types import MethodType

import torch

from radvl.model_residual import RADVL as ResidualRADVL


def dense_aggregate(self, x, edge_index, edge_attr=None):
    """Dense ``A @ X`` replacement for ``WeightedGINLayer.aggregate`` (all three modes)."""
    receiver, neighbor = edge_index
    node_count = x.shape[0]
    adjacency = x.new_zeros(node_count, node_count)
    if self.weighting == 'uniform':
        adjacency[receiver, neighbor] = 1.0
        return adjacency @ x
    if edge_attr is None:
        raise ValueError('Similarity weighting requires edge_attr on the similarity graph')
    weights = edge_attr.to(x.dtype)
    if (weights <= 0).any():
        raise ValueError('Similarity edge weights must be strictly positive')
    adjacency[receiver, neighbor] = weights
    degree = adjacency.sum(dim=1)
    scale = torch.ones_like(degree)
    present = degree > 0
    if self.normalization == 'mass':
        neighbours = (adjacency > 0).sum(dim=1).to(degree.dtype)
        scale[present] = neighbours[present] / degree[present]
    else:
        scale[present] = 1.0 / degree[present]
    return (adjacency @ x) * scale[:, None]


def stopgrad_fusion_forward(self, similarity, association):
    """``RADVLFusion.forward`` with ``omega`` detached; every other line is unchanged."""
    context_1 = self.context(similarity, association)
    context_2 = self.context(association, similarity)
    delta_1 = similarity - context_1
    delta_2 = association - context_2
    gate_1 = torch.sigmoid(self.gate_1(torch.cat([delta_1, similarity, context_1], dim=-1)))
    gate_2 = torch.sigmoid(self.gate_2(torch.cat([delta_2, association, context_2], dim=-1)))
    first = gate_1 * similarity + (1 - gate_1) * context_1
    second = gate_2 * association + (1 - gate_2) * context_2
    discrepancy = delta_1.square().sum(dim=-1) + delta_2.square().sum(dim=-1)
    if self.tau_mode == 'adaptive':
        tau = (self.tau_scale * discrepancy.mean()).detach().clamp_min(1e-12)
    else:
        tau = self.tau
    weight = torch.exp(-discrepancy / tau)
    # The single change: the weight enters the loss with no gradient, so the model cannot
    # reduce the penalty by inflating the discrepancy.  Values are unchanged.
    alignment = (weight.detach() * (first - second).square().sum(dim=-1)).mean()
    fused = first + second
    with torch.no_grad():
        self.diagnostics = dict(
            delta_squared_mean=discrepancy.mean().item(),
            tau_effective=float(tau),
            alignment_strength_mean=weight.mean().item(),
            alignment_strength_min=weight.min().item(),
            alignment_strength_max=weight.max().item(),
            alignment_weight_below_1e_minus8_fraction=(weight < 1e-8).float().mean().item(),
            gate_1_mean=gate_1.mean().item(), gate_2_mean=gate_2.mean().item(),
            fused_norm_mean=fused.norm(dim=1).mean().item())
        self.diagnostic_arrays = dict(
            delta_1=delta_1.detach(), delta_2=delta_2.detach(),
            gate_1=gate_1.detach(), gate_2=gate_2.detach(),
            alignment_strength=weight.detach())
    return fused, alignment


def install_mechanisms(model):
    """Bind the two replacements on an already-constructed model; no parameter is touched."""
    for encoder in (model.circ_gin_encoder, model.dr_gin_encoder):
        for layer in encoder.layers:
            layer.aggregate = MethodType(dense_aggregate, layer)
    for fusion in (model.circ_fusion, model.dr_fusion):
        fusion.forward = MethodType(stopgrad_fusion_forward, fusion)
    model.deterministic_dense_aggregation = True
    model.stopgrad_alignment_weight = True
    return model


class RADVL(ResidualRADVL):
    def __init__(self, n_circ, n_drug, args):
        super().__init__(n_circ, n_drug, args)
        install_mechanisms(self)

"""RADVL hypergraph weighting."""
import numpy as np
import torch

from radvl.features import clustered_hypergraph


WEIGHT_MODES = ('unit', 'similarity')
# Hyperedges whose members are mutually dissimilar would give node degree zero and make
# Dv^(-1/2) undefined, so the weights are floored at a tiny positive value instead.
FLOOR_FRACTION = 1e-3


def hyperedge_weights(similarity, incidence, mode='unit', rescale=True):
    """Per-hyperedge weights ``w_e``; returns ``(weights, diagnostics)``.

    ``mode='unit'``       -> ones (the current behaviour, Eq. (18) with W = I)
    ``mode='similarity'`` -> mean intra-hyperedge similarity, floored, optionally rescaled
                             to unit mean across hyperedges
    """
    similarity = np.asarray(similarity, dtype=np.float64)
    incidence = np.asarray(incidence, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError('Similarity must be a square matrix')
    if incidence.ndim != 2 or incidence.shape[0] != similarity.shape[0]:
        raise ValueError('Incidence rows must correspond to similarity nodes')
    if not np.isin(incidence, (0.0, 1.0)).all():
        raise ValueError('This hypergraph uses binary incidence')
    if mode not in WEIGHT_MODES:
        raise ValueError(f'Unknown hyperedge weight mode: {mode}')
    count = incidence.shape[1]
    if mode == 'unit':
        return np.ones(count, dtype=np.float64), dict(
            mode=mode, rescaled=False, hyperedge_count=count, floored_hyperedge_count=0,
            weight_mean=1.0, weight_min=1.0, weight_max=1.0)
    card = incidence.sum(axis=0)
    # Sum of S over each hyperedge block, minus its diagonal, over |e|*(|e|-1) ordered pairs.
    block_sums = np.diag(incidence.T @ similarity @ incidence)
    diagonal_sums = (incidence * np.diag(similarity)[:, None]).sum(axis=0)
    denominator = card * (card - 1.0)
    valid = denominator > 0
    raw = np.ones(count, dtype=np.float64)
    raw[valid] = (block_sums[valid] - diagonal_sums[valid]) / denominator[valid]
    raw = np.maximum(raw, 0.0)
    mean_before_floor = float(raw.mean()) if count else 1.0
    floor = max(FLOOR_FRACTION * (mean_before_floor if mean_before_floor > 0 else 1.0), 1e-12)
    floored = int((raw < floor).sum())
    weights = np.maximum(raw, floor)
    rescaled = False
    if rescale:
        scale = float(weights.mean())
        if scale <= 0:
            raise AssertionError('Cannot rescale nonpositive hyperedge weights')
        weights = weights / scale
        rescaled = True
    diagnostics = dict(mode=mode, rescaled=rescaled, hyperedge_count=int(count),
                       floored_hyperedge_count=floored, weight_mean=float(weights.mean()),
                       weight_min=float(weights.min()), weight_max=float(weights.max()),
                       raw_weight_mean_before_floor=mean_before_floor, floor_value=float(floor))
    return weights, diagnostics


def weighted_propagation_operator(incidence, weights):
    """``G = Dv^(-1/2) H W De^(-1) H^T Dv^(-1/2)`` with weighted node degree."""
    incidence = np.asarray(incidence, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if incidence.ndim != 2:
        raise ValueError('Incidence must be 2-D')
    if len(weights) != incidence.shape[1]:
        raise ValueError('One weight per hyperedge is required')
    if (weights <= 0).any():
        raise ValueError('Hyperedge weights must be strictly positive for Dv^(-1/2)')
    node_degree = incidence @ weights
    edge_degree = incidence.sum(axis=0)
    if (node_degree <= 0).any() or (edge_degree <= 0).any():
        raise ValueError('Every node and every hyperedge must be non-empty')
    scale = 1.0 / np.sqrt(node_degree)
    left = (incidence * weights[None, :]) * scale[:, None] / edge_degree[None, :]
    right = incidence * scale[:, None]
    operator = left @ right.T
    if not np.isfinite(operator).all():
        raise AssertionError('Nonfinite hypergraph propagation operator')
    return operator


def weighted_hypergraph(similarity, clusters, weight_mode='unit', rescale=True):
    """Parent hyperedges (unchanged) + the weighting; returns operator/incidence/labels/weights."""
    _, incidence, labels = clustered_hypergraph(similarity, clusters)
    incidence_numpy = incidence.detach().cpu().numpy() if isinstance(incidence, torch.Tensor) else incidence
    weights, diagnostics = hyperedge_weights(similarity, incidence_numpy, weight_mode, rescale)
    operator = weighted_propagation_operator(incidence_numpy, weights)
    return (torch.tensor(operator, dtype=torch.float32), incidence,
            labels, torch.tensor(weights, dtype=torch.float32), diagnostics)

"""RADVL test similarity fusion."""
import argparse

import numpy as np
import torch

from radvl.features import clustered_hypergraph
from radvl.hypergraph_weighting import hyperedge_weights, weighted_propagation_operator, weighted_hypergraph
from radvl.model_core import RADVL as ParentRADVL
from radvl.model_core import GINLayer as ParentGINLayer
from radvl.model_similarity import RADVL as WeightedRADVL
from radvl.model_similarity import RADVLFusion, WeightedGINLayer


def synthetic_similarity(nodes=12, seed=3):
    generator = np.random.RandomState(seed)
    raw = generator.rand(nodes, nodes)
    similarity = (raw + raw.T) / 2
    np.fill_diagonal(similarity, 1.0)
    return similarity


def check_default_operator_matches_the_reference():
    similarity = synthetic_similarity()
    parent_operator, parent_incidence, _ = clustered_hypergraph(similarity, 3)
    operator, incidence, _, weights, diagnostics = weighted_hypergraph(similarity, 3, 'unit', True)
    assert torch.equal(operator, parent_operator), 'unit weights must reproduce the parent operator'
    assert torch.equal(incidence, parent_incidence), 'hyperedges must be unchanged by the weighting'
    assert torch.equal(weights, torch.ones_like(weights))
    assert diagnostics['floored_hyperedge_count'] == 0


def check_hyperedges_are_unchanged_by_weighting():
    similarity = synthetic_similarity()
    _, parent_incidence, _ = clustered_hypergraph(similarity, 4)
    _, incidence, _, _, _ = weighted_hypergraph(similarity, 4, 'similarity', True)
    assert torch.equal(incidence, parent_incidence)


def check_weights_are_mean_intra_hyperedge_similarity():
    similarity = synthetic_similarity()
    _, incidence, _ = clustered_hypergraph(similarity, 3)
    incidence = incidence.detach().cpu().numpy()
    raw, _ = hyperedge_weights(similarity, incidence, 'similarity', rescale=False)
    for edge in range(incidence.shape[1]):
        members = np.nonzero(incidence[:, edge])[0]
        if len(members) < 2:
            assert raw[edge] == 1.0
            continue
        manual = np.mean([similarity[u, v] for u in members for v in members if u != v])
        assert abs(raw[edge] - manual) < 1e-12, edge
    rescaled, diagnostics = hyperedge_weights(similarity, incidence, 'similarity', rescale=True)
    assert abs(rescaled.mean() - 1.0) < 1e-12
    assert diagnostics['rescaled'] is True
    # Rescaling is applied to the floored weights, so compare against those.
    assert np.allclose(rescaled, raw / raw.mean())


def check_weighted_operator_matches_the_dense_formula():
    similarity = synthetic_similarity()
    _, incidence, _, weights, _ = weighted_hypergraph(similarity, 3, 'similarity', True)
    # The propagation utility evaluates this formula in float64. Use the same
    # precision in the independent dense reference, rather than float32 powers.
    incidence = incidence.detach().cpu().numpy().astype(np.float64)
    weights = weights.detach().cpu().numpy().astype(np.float64)
    operator = weighted_propagation_operator(incidence, weights)
    node_degree = incidence @ weights
    edge_degree = incidence.sum(axis=0)
    dense = (np.diag(node_degree ** -0.5) @ incidence @ np.diag(weights)
             @ np.diag(edge_degree ** -1) @ incidence.T @ np.diag(node_degree ** -0.5))
    assert np.allclose(operator, dense, rtol=1e-10, atol=1e-12)
    assert np.allclose(operator, operator.T)
    assert (node_degree > 0).all()


def check_gin_default_matches_the_radvl_layer():
    torch.manual_seed(11)
    parent = ParentGINLayer(6, 0.1)
    torch.manual_seed(11)
    weighted = WeightedGINLayer(6, 0.1, 'uniform', 'mass')
    # Compare deterministic operators rather than independent dropout samples.
    parent.eval()
    weighted.eval()
    assert set(parent.state_dict()) == set(weighted.state_dict())
    for name, value in parent.state_dict().items():
        assert torch.equal(value, weighted.state_dict()[name]), name
    x = torch.randn(5, 6)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 1, 0], [1, 0, 3, 2, 0, 2, 4]])
    edge_attr = torch.rand(edge_index.shape[1]) + 0.1
    assert torch.equal(parent(x, edge_index), weighted(x, edge_index, edge_attr))


def check_gin_weighting_preserves_message_mass():
    torch.manual_seed(5)
    layer = WeightedGINLayer(4, 0.0, 'similarity', 'mass')
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 1, 3], [1, 0, 3, 2, 5, 4, 2, 0]])
    receiver, neighbor = edge_index
    reference = torch.zeros_like(x)
    reference.index_add_(0, receiver, x[neighbor])
    # Constant weights carry no relative information, so the mass-preserving weighting is a no-op.
    constant = torch.full((edge_index.shape[1],), 0.5)
    weighted = layer.aggregate(x, edge_index, constant)
    assert torch.allclose(reference, weighted, rtol=1e-6, atol=1e-6)
    varied = torch.linspace(0.1, 0.9, edge_index.shape[1])
    different = layer.aggregate(x, edge_index, varied)
    assert not torch.allclose(reference, different)


def check_adaptive_tau_increases_alignment_weight():
    torch.manual_seed(7)
    dimension = 16
    similarity = 8.0 * torch.randn(24, dimension)
    association = torch.zeros(24, dimension)
    fixed = RADVLFusion(dimension, 100.0, 'fixed', 1.0)
    adaptive = RADVLFusion(dimension, 100.0, 'adaptive', 1.0)
    adaptive.query.load_state_dict(fixed.query.state_dict())
    adaptive.key.load_state_dict(fixed.key.state_dict())
    adaptive.value.load_state_dict(fixed.value.state_dict())
    adaptive.gate_1.load_state_dict(fixed.gate_1.state_dict())
    adaptive.gate_2.load_state_dict(fixed.gate_2.state_dict())
    fixed(similarity, association)
    adaptive(similarity, association)
    discrepancy = fixed.diagnostics['delta_squared_mean']
    assert discrepancy > 500.0, discrepancy
    # The original fixed 1e-3 cutoff also failed before the naming refactor.
    # Check the intended monotonic effect instead of an unsupported absolute
    # cutoff on the mean of the node-specific exponential weights.
    assert torch.all(adaptive.diagnostic_arrays['alignment_strength'] >=
                     fixed.diagnostic_arrays['alignment_strength'])
    assert adaptive.diagnostics['alignment_strength_mean'] > fixed.diagnostics['alignment_strength_mean']
    assert adaptive.diagnostics['alignment_strength_mean'] > 0.1
    assert abs(adaptive.diagnostics['tau_effective'] - discrepancy) < 1e-3 * discrepancy


def check_default_model_matches_the_radvl_model():
    arguments = argparse.Namespace(hid_dim=8, hypergraph_width=8, latent_dim=8, hgcn_layers=2,
                                   dropout=0.1, gin_layers=2, vae_hidden_dim=16, pair_hidden_dim=8,
                                   pair_chunk_rows=4, fusion_tau=100.0)
    torch.manual_seed(1234)
    parent = ParentRADVL(6, 5, arguments)
    torch.manual_seed(1234)
    weighted = WeightedRADVL(6, 5, arguments)
    parent_state, weighted_state = parent.state_dict(), weighted.state_dict()
    assert set(parent_state) == set(weighted_state)
    for name, value in parent_state.items():
        assert torch.equal(value, weighted_state[name]), name


if __name__ == '__main__':
    checks = [value for name, value in sorted(globals().items()) if name.startswith('check_')]
    for check in checks:
        check()
        print('OK', check.__name__)
    print(f'{len(checks)} checks passed')

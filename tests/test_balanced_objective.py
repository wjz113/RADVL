"""RADVL test balanced objective."""
import argparse

import torch

from radvl.objectives import radvl_objective
from radvl.objectives_pair_scaling import radvl_objective_pairscaled
from radvl.objectives_balanced import radvl_objective_balanced
from radvl.model_similarity import RADVLFusion, WeightedGINLayer
from radvl.model_residual import RADVL as ResidualRADVL
from radvl.model_balanced import RADVL as BalancedRADVL, dense_aggregate, stopgrad_fusion_forward


class FakeModel:
    def __init__(self, n_circ=5, n_drug=4, latent=3, seed=7):
        generator = torch.Generator().manual_seed(seed)
        self.prediction_logits = torch.randn(n_circ, n_drug, generator=generator)
        self.z_node_mean = torch.randn(n_circ, latent, generator=generator)
        self.z_node_log_var = 0.1 * torch.randn(n_circ, latent, generator=generator)
        self.z_edge_mean = torch.randn(n_drug, latent, generator=generator)
        self.z_edge_log_var = 0.1 * torch.randn(n_drug, latent, generator=generator)


def loss_arguments(lambda_scale=None):
    values = dict(lambda_av=0.37, lambda_align=2.5, lambda_cov=0.5, eta_pos=1.0, eta_neg=1.0)
    if lambda_scale is not None:
        values['lambda_scale'] = lambda_scale
    return argparse.Namespace(**values)


def loss_fixture():
    model = FakeModel()
    generator = torch.Generator().manual_seed(11)
    positive = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])
    negative = torch.tensor([[0, 1], [1, 0], [4, 2], [3, 0], [2, 3]])
    reconstruction = torch.randn(5, 4, generator=generator)
    return model, reconstruction, positive, negative, torch.tensor(0.25), torch.tensor(0.5)


def model_arguments():
    return argparse.Namespace(hid_dim=8, hypergraph_width=8, latent_dim=8, hgcn_layers=2,
                              dropout=0.1, gin_layers=2, vae_hidden_dim=16, pair_hidden_dim=8,
                              pair_chunk_rows=4, fusion_tau=100.0, fusion_tau_scale=1.0,
                              prediction_head='residual_mlp')


def check_mean_form_is_the_pair_scaled_form_over_omega():
    model, reconstruction, positive, negative, covariance, alignment = loss_fixture()
    pair_count = len(positive) + len(negative)
    args = loss_arguments()
    balanced, _ = radvl_objective_balanced(model, reconstruction, positive, negative,
                                                covariance, alignment, args)
    scaled, _ = radvl_objective_pairscaled(model, reconstruction, positive, negative,
                                                covariance, alignment, loss_arguments('pairs'))
    summed, _ = radvl_objective(model, reconstruction, positive, negative,
                                     covariance, alignment, args)
    assert torch.allclose(balanced, scaled / pair_count, rtol=1e-6, atol=1e-8)
    # The auxiliaries keep their manuscript weights, so this is NOT the summed form rescaled.
    assert not torch.allclose(balanced, summed / pair_count, rtol=1e-6, atol=1e-8)


def check_dense_aggregation_matches_the_index_add_version():
    torch.manual_seed(5)
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 1, 3], [1, 0, 3, 2, 5, 4, 2, 0]])
    edge_attr = torch.linspace(0.2, 0.9, edge_index.shape[1])
    uniform = WeightedGINLayer(4, 0.0, 'uniform', 'mass')
    reference = uniform.aggregate(x, edge_index, None)
    dense = dense_aggregate(uniform, x, edge_index, None)
    assert torch.allclose(reference, dense, rtol=1e-6, atol=1e-6)
    for normalization in ('mass', 'mean'):
        weighted = WeightedGINLayer(4, 0.0, 'similarity', normalization)
        reference = weighted.aggregate(x, edge_index, edge_attr)
        dense = dense_aggregate(weighted, x, edge_index, edge_attr)
        assert torch.allclose(reference, dense, rtol=1e-6, atol=1e-6), normalization


def check_stopgrad_keeps_the_value_but_removes_the_discrepancy_gradient():
    torch.manual_seed(7)
    dimension, nodes = 16, 24
    fusion = RADVLFusion(dimension, 100.0, 'adaptive', 1.0)
    similarity = 8.0 * torch.randn(nodes, dimension)
    association = torch.randn(nodes, dimension)

    def run(path):
        sin = similarity.clone().requires_grad_(True)
        ain = association.clone().requires_grad_(True)
        _, alignment = path(fusion, sin, ain)
        alignment.backward()
        return float(alignment.detach()), sin.grad.clone(), ain.grad.clone()

    parent_value, parent_sim_grad, parent_asso_grad = run(lambda f, s, a: f(s, a))
    detached_value, detached_sim_grad, detached_asso_grad = run(stopgrad_fusion_forward)
    # Same loss value and same diagnostics, different gradient path.
    assert parent_value == detached_value
    assert not torch.allclose(parent_sim_grad, detached_sim_grad)
    # The alignment still trains the two gated representations.
    assert detached_asso_grad.abs().max() > 0.0
    assert detached_sim_grad.abs().max() > 0.0


def check_mechanisms_are_parameter_free():
    arguments = model_arguments()
    torch.manual_seed(1234)
    residual = ResidualRADVL(6, 5, arguments)
    torch.manual_seed(1234)
    balanced = BalancedRADVL(6, 5, arguments)
    residual_state, balanced_state = residual.state_dict(), balanced.state_dict()
    assert set(residual_state) == set(balanced_state)
    for name, value in residual_state.items():
        assert torch.equal(value, balanced_state[name]), name
    assert sum(p.numel() for p in residual.parameters()) == sum(p.numel() for p in balanced.parameters())
    assert balanced.deterministic_dense_aggregation is True
    assert balanced.stopgrad_alignment_weight is True
    # The instance-level bindings really took effect.
    for encoder in (balanced.circ_gin_encoder, balanced.dr_gin_encoder):
        for layer in encoder.layers:
            assert layer.aggregate.__func__ is dense_aggregate
    for fusion in (balanced.circ_fusion, balanced.dr_fusion):
        assert fusion.forward.__func__ is stopgrad_fusion_forward


if __name__ == '__main__':
    checks = [value for name, value in sorted(globals().items()) if name.startswith('check_')]
    for check in checks:
        check()
        print('OK', check.__name__)
    print(f'{len(checks)} checks passed')

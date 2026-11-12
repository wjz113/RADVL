"""RADVL test residual prediction."""
import argparse

import torch
from torch import nn

from radvl.objectives import radvl_objective
from radvl.model_similarity import RADVL as A3RADVL
from radvl.model_residual import RADVL as ResidualRADVL


class FakeGraph:
    """Carries the two attributes the GIN encoder reads."""

    def __init__(self, x, edge_index, edge_attr=None):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


def model_arguments(prediction_head):
    return argparse.Namespace(hid_dim=8, hypergraph_width=8, latent_dim=8, hgcn_layers=2,
                              dropout=0.1, gin_layers=2, vae_hidden_dim=16, pair_hidden_dim=8,
                              pair_chunk_rows=4, fusion_tau=100.0, fusion_tau_scale=1.0,
                              prediction_head=prediction_head,
                              lambda_av=1.0, lambda_align=2.0, lambda_cov=0.5,
                              eta_pos=1.0, eta_neg=1.0)


def build_inputs(seed=21):
    torch.manual_seed(seed)
    n_circ, n_drug = 6, 5
    circ_x = 0.5 * torch.rand(n_circ, n_circ)
    circ_x = (circ_x + circ_x.T) / 2
    drug_x = 0.5 * torch.rand(n_drug, n_drug)
    drug_x = (drug_x + drug_x.T) / 2
    circ_graph = FakeGraph(circ_x, torch.tensor([[0, 1, 2, 3, 4, 5, 1, 2], [1, 0, 3, 2, 5, 4, 2, 0]]))
    drug_graph = FakeGraph(drug_x, torch.tensor([[0, 1, 2, 3, 4, 1], [1, 0, 3, 2, 1, 4]]))
    A = (torch.rand(n_circ, n_drug) > 0.5).float()
    return A, circ_graph, drug_graph, torch.eye(n_circ), torch.eye(n_drug)


def forward_once(model, inputs=None):
    inputs = build_inputs() if inputs is None else inputs
    return model(*inputs)


def check_mlp_mode_matches_the_a3_model():
    arguments = model_arguments('mlp')
    torch.manual_seed(1234)
    a3 = A3RADVL(6, 5, arguments)
    torch.manual_seed(1234)
    residual = ResidualRADVL(6, 5, arguments)
    a3_state, residual_state = a3.state_dict(), residual.state_dict()
    assert set(a3_state) == set(residual_state), sorted(set(a3_state) ^ set(residual_state))
    for name, value in a3_state.items():
        assert torch.equal(value, residual_state[name]), name


def check_residual_starts_from_the_association_score():
    model = ResidualRADVL(6, 5, model_arguments('residual_mlp'))
    assert float(model.residual_scale.detach()) == 0.0
    forward_once(model)
    assert torch.equal(model.prediction_logits, model.association_logits())
    assert float(model.residual_gamma) == 0.0


def check_residual_combination_is_exact():
    model = ResidualRADVL(6, 5, model_arguments('residual_mlp'))
    model.residual_scale.data.fill_(0.5)
    forward_once(model)
    expected = model.association_logits() + torch.tanh(torch.tensor(0.5)) * model.fusion_logits
    assert torch.allclose(model.prediction_logits, expected, rtol=1e-6, atol=1e-6)
    assert abs(float(model.residual_gamma) - float(torch.tanh(torch.tensor(0.5)))) < 1e-6
    assert not torch.allclose(model.prediction_logits, model.fusion_logits)


def check_fusion_head_is_frozen_at_initialisation():
    model = ResidualRADVL(6, 5, model_arguments('residual_mlp'))
    arguments = model_arguments('residual_mlp')
    scores, reconstruction, drug_fused, circ_fused, covariance, alignment = forward_once(model)
    positive = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    negative = torch.tensor([[0, 1], [1, 0], [5, 2], [3, 0], [2, 3]])
    total, _ = radvl_objective(model, reconstruction, positive, negative,
                                    covariance, alignment, arguments)
    total.backward()
    # The residual parameter itself must receive gradient (that is the warm-up path) ...
    assert model.residual_scale.grad is not None
    assert torch.isfinite(model.residual_scale.grad).all()
    assert float(model.residual_scale.grad.abs()) > 0.0
    # ... while the fusion head receives exactly none, because gamma = 0 ...
    for name, parameter in model.pair_predictor.named_parameters():
        if parameter.grad is not None:
            assert float(parameter.grad.abs().max()) == 0.0, name
    # ... and the fusion modules still do, through the alignment loss.
    alignment_grads = [parameter.grad for parameter in model.circ_fusion.parameters() if parameter.grad is not None]
    assert alignment_grads
    assert max(float(gradient.abs().max()) for gradient in alignment_grads) > 0.0


def check_diagnostic_branches_are_exposed():
    residual = ResidualRADVL(6, 5, model_arguments('residual_mlp'))
    forward_once(residual)
    assert {'association', 'fused_inner_product', 'combined',
            'fusion_only', 'association_plus_fusion'} <= set(residual.diagnostic_scores)
    for name, value in residual.diagnostic_scores.items():
        assert not value.requires_grad, name
    plain = ResidualRADVL(6, 5, model_arguments('mlp'))
    forward_once(plain)
    assert 'fusion_only' not in plain.diagnostic_scores


if __name__ == '__main__':
    checks = [value for name, value in sorted(globals().items()) if name.startswith('check_')]
    for check in checks:
        check()
        print('OK', check.__name__)
    print(f'{len(checks)} checks passed')

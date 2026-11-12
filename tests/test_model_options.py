"""RADVL test model options."""
import argparse
import math

import torch

from radvl.objectives_balanced import radvl_objective_balanced
from radvl.objectives_variants import coefficient_equivalent, radvl_objective_controls
from radvl.model_balanced import RADVL as BalancedRADVL
from radvl.model_variants import RADVL as ControlsRADVL


class FakeGraph:
    def __init__(self, x, edge_index, edge_attr=None):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


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


PAIRS = dict(positive=torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]),
             negative=torch.tensor([[0, 1], [1, 0], [5, 2], [3, 0], [2, 3]]))


def model_arguments(**overrides):
    values = dict(hid_dim=8, hypergraph_width=8, latent_dim=8, hgcn_layers=2, dropout=0.0,
                  gin_layers=2, vae_hidden_dim=16, pair_hidden_dim=8, pair_chunk_rows=4,
                  fusion_tau=100.0, fusion_tau_scale=1.0, prediction_head='residual_mlp',
                  residual_init=0.0, association_gradient='on', similarity_diagnostic='on',
                  lambda_av=0.01, lambda_align=0.02, lambda_cov=10.0, lambda_rec=None,
                  beta_kl=None, lambda_cov_star=None, loss_coefficients='grouped',
                  eta_pos=1.0, eta_neg=1.0)
    values.update(overrides)
    return argparse.Namespace(**values)


def forward_fixture(arguments, model_class=ControlsRADVL, seed=1234, forward_seed=99):
    torch.manual_seed(seed)
    model = model_class(6, 5, arguments)
    inputs = build_inputs()
    torch.manual_seed(forward_seed)
    scores, reconstruction, drug_fused, circ_fused, covariance, alignment = model(*inputs)
    return model, (scores, reconstruction, drug_fused, circ_fused, covariance, alignment)


def parts_of(model, outputs, arguments, objective):
    _, reconstruction, _, _, covariance, alignment = outputs
    return objective(model, reconstruction, PAIRS['positive'], PAIRS['negative'],
                     covariance, alignment, arguments)


def check_grouped_mode_is_bit_identical_to_the_balanced_objective():
    arguments = model_arguments()
    model, outputs = forward_fixture(arguments)
    grouped_total, grouped = parts_of(model, outputs, arguments, radvl_objective_controls)
    balanced_total, balanced = parts_of(model, outputs, arguments, radvl_objective_balanced)
    assert torch.equal(grouped_total, balanced_total)
    shared = set(grouped) & set(balanced)
    assert {'prediction_bce', 'weighted_reconstruction', 'weighted_kl', 'weighted_covariance_term',
            'weighted_association_view', 'weighted_alignment'} <= shared
    for name in shared:
        left, right = grouped[name], balanced[name]
        if torch.is_tensor(left):
            assert torch.equal(left, right), name


def check_decoupled_mode_is_equivalent_with_equivalent_coefficients():
    arguments = model_arguments()
    equivalent = coefficient_equivalent(arguments)
    assert equivalent == dict(lambda_rec=0.01, beta_kl=0.01, lambda_cov_star=0.1)
    # Requesting the decoupled mode without coefficients must fail loudly.
    missing = model_arguments(loss_coefficients='decoupled')
    missing_model, missing_outputs = forward_fixture(missing)
    try:
        parts_of(missing_model, missing_outputs, missing, radvl_objective_controls)
    except ValueError:
        pass
    else:
        raise AssertionError('decoupled mode must require the three coefficients')
    decoupled_arguments = model_arguments(loss_coefficients='decoupled', **equivalent)
    model, outputs = forward_fixture(decoupled_arguments)
    decoupled_total, decoupled = parts_of(model, outputs, decoupled_arguments,
                                          radvl_objective_controls)
    grouped_arguments = model_arguments()
    _, grouped = parts_of(model, outputs, grouped_arguments, radvl_objective_controls)
    assert math.isclose(float(decoupled_total), float(grouped['prediction_bce']
                        + grouped['weighted_association_view'] + grouped['weighted_alignment']),
                        rel_tol=1e-6)
    for name in ('prediction_bce', 'reconstruction_bce', 'kl_standard', 'covariance_raw',
                 'alignment_raw'):
        assert torch.allclose(decoupled[name], grouped[name], rtol=1e-6, atol=1e-8), name
    total = (decoupled['weighted_reconstruction'] + decoupled['weighted_kl']
             + decoupled['weighted_covariance_term'])
    assert torch.allclose(total, decoupled['weighted_association_view'], rtol=1e-6, atol=1e-8)
    assert torch.allclose(decoupled['weighted_covariance_term'],
                          0.1 * decoupled['covariance_raw'], rtol=1e-6, atol=1e-8)
    assert torch.allclose(decoupled['weighted_reconstruction'],
                          0.01 * decoupled['reconstruction_bce'], rtol=1e-6, atol=1e-8)
    assert torch.allclose(decoupled['weighted_kl'], 0.01 * decoupled['kl_standard'],
                          rtol=1e-6, atol=1e-8)


def check_switches_do_not_change_the_parameters_or_the_forward():
    arguments = model_arguments()
    # The reference is the BALANCED model: that is the model the audit entry runs, so the control
    # entry must inherit the dense aggregation and the detached alignment weight from it.
    baseline, baseline_outputs = forward_fixture(arguments, model_class=BalancedRADVL)
    controls, controls_outputs = forward_fixture(arguments)
    assert baseline.deterministic_dense_aggregation is True
    assert baseline.stopgrad_alignment_weight is True
    assert controls.deterministic_dense_aggregation is True
    assert controls.stopgrad_alignment_weight is True
    left, right = baseline.state_dict(), controls.state_dict()
    assert set(left) == set(right)
    for name, value in left.items():
        assert torch.equal(value, right[name]), name
    assert torch.equal(baseline_outputs[0], controls_outputs[0])
    assert torch.equal(baseline.prediction_logits, controls.prediction_logits)
    assert sum(p.numel() for p in baseline.parameters()) == sum(p.numel() for p in controls.parameters())


def check_stopgrad_cuts_the_direct_path_to_the_covae():
    def prediction_gradients(arguments):
        model, outputs = forward_fixture(arguments)
        _, parts = parts_of(model, outputs, arguments, radvl_objective_controls)
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        gradients = torch.autograd.grad(parts['prediction_bce'], [p for _, p in named],
                                        retain_graph=True, allow_unused=True)
        return model, parts, {name: gradient for (name, _), gradient in zip(named, gradients)}

    def magnitude(gradients):
        return sum(float(g.detach().pow(2).sum()) for g in gradients.values()
                   if g is not None) ** 0.5

    # gamma = 0 at initialisation: stopping the association score leaves exactly one parameter the
    # prediction loss can still move -- the residual scale itself, because dlogit/da = s_fusion.
    # Every other parameter must have an exactly zero gradient.
    on_model, on_parts, on_gradients = prediction_gradients(
        model_arguments(association_gradient='on'))
    off_model, off_parts, off_gradients = prediction_gradients(
        model_arguments(association_gradient='off'))
    assert torch.equal(on_parts['prediction_bce'], off_parts['prediction_bce'])
    assert torch.equal(on_model.prediction_logits, off_model.prediction_logits)
    movable = {name for name, gradient in off_gradients.items()
               if gradient is not None and float(gradient.abs().max()) > 0.0}
    assert movable == {'residual_scale'}, sorted(movable)
    assert magnitude(off_gradients) > 0.0
    assert magnitude(on_gradients) > magnitude(off_gradients)
    assert torch.equal(on_model.prediction_logits, on_model.association_logits())

    # gamma > 0: the indirect path through s_fusion survives, but it is weaker than the direct one.
    on_model, _, on_gradients = prediction_gradients(
        model_arguments(residual_init=1.0, association_gradient='on'))
    off_model, _, off_gradients = prediction_gradients(
        model_arguments(residual_init=1.0, association_gradient='off'))
    assert torch.equal(on_model.prediction_logits, off_model.prediction_logits)
    on_norm = magnitude(on_gradients)
    off_norm = magnitude(off_gradients)
    assert 0.0 < off_norm < on_norm, (on_norm, off_norm)


def check_similarity_diagnostic_branch_is_observer_only():
    arguments = model_arguments()
    model, outputs = forward_fixture(arguments)
    assert 'similarity_inner_product' in model.diagnostic_scores
    branch = model.diagnostic_scores['similarity_inner_product']
    assert branch.shape == (6, 5)
    assert not branch.requires_grad
    expected = torch.sigmoid(model.circ_similarity_view @ model.drug_similarity_view.T)
    assert torch.allclose(branch, expected, rtol=1e-6, atol=1e-7)
    assert model.circ_similarity_view.shape == (6, arguments.hid_dim)
    assert model.drug_similarity_view.shape == (5, arguments.hid_dim)
    plain, plain_outputs = forward_fixture(model_arguments(similarity_diagnostic='off'))
    assert 'similarity_inner_product' not in plain.diagnostic_scores
    assert torch.equal(plain_outputs[0], outputs[0])
    assert torch.equal(plain.prediction_logits, model.prediction_logits)


def check_alignment_warmup_schedule():
    from radvl.objectives_alignment import ScheduledAlignmentObjective

    arguments = model_arguments(lambda_align_warmup=5)
    model, outputs = forward_fixture(arguments)
    objective = ScheduledAlignmentObjective(radvl_objective_controls)
    factors, ratios = [], []
    for _ in range(7):
        _, parts = parts_of(model, outputs, arguments, objective)
        factors.append(model.alignment_schedule_factor)
        ratios.append(float(parts['weighted_alignment']) / float(parts['alignment_raw']))
    assert factors == [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0], factors
    assert all(abs(ratio - 0.02 * factor) < 1e-9 for ratio, factor in zip(ratios, factors)), ratios
    # The caller's namespace must never be mutated.
    assert arguments.lambda_align == 0.02
    # A new fold (new model object) restarts the schedule.
    other_arguments = model_arguments(lambda_align_warmup=5)
    other, other_outputs = forward_fixture(other_arguments, seed=4321)
    _, _ = parts_of(other, other_outputs, other_arguments, objective)
    assert other.alignment_schedule_factor == 0.2
    # warm-up 0 leaves the objective untouched.
    flat_arguments = model_arguments()
    plain = ScheduledAlignmentObjective(radvl_objective_controls)
    flat_model, flat_outputs = forward_fixture(flat_arguments)
    _, flat_parts = parts_of(flat_model, flat_outputs, flat_arguments, plain)
    assert flat_model.alignment_schedule_factor == 1.0
    assert abs(float(flat_parts['weighted_alignment'])
               - 0.02 * float(flat_parts['alignment_raw'])) < 1e-9


if __name__ == '__main__':
    checks = [value for name, value in sorted(globals().items()) if name.startswith('check_')]
    for check in checks:
        check()
        print('OK', check.__name__)
    print(f'{len(checks)} checks passed')

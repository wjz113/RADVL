"""RADVL test gradient diagnostics."""
import argparse
import copy
import math

import numpy as np
import torch
from torch import nn

from radvl.gradient_diagnostics import AGGREGATE_LABEL, GradientAuditObjective, latent_geometry, parameter_groups, run_audit
from radvl.objectives import covariance_penalty
from radvl.objectives_balanced import radvl_objective_balanced
from radvl.model_balanced import RADVL


class FakeGraph:
    def __init__(self, x, edge_index, edge_attr=None):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


def model_arguments(prediction_head='residual_mlp'):
    return argparse.Namespace(hid_dim=8, hypergraph_width=8, latent_dim=8, hgcn_layers=2,
                              dropout=0.0, gin_layers=2, vae_hidden_dim=16, pair_hidden_dim=8,
                              pair_chunk_rows=4, fusion_tau=100.0, fusion_tau_scale=1.0,
                              prediction_head=prediction_head, residual_init=0.0,
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


PAIRS = dict(positive=torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]),
             negative=torch.tensor([[0, 1], [1, 0], [5, 2], [3, 0], [2, 3]]))


def run_objective(model, objective, arguments):
    inputs = build_inputs()
    torch.manual_seed(99)
    _, reconstruction, _, _, covariance, alignment = model(*inputs)
    return objective(model, reconstruction, PAIRS['positive'], PAIRS['negative'],
                     covariance, alignment, arguments)


class FakeAuditModel(nn.Module):
    """Named so that the group mapping has something to find, including one unassigned module.

    The latent means and the prediction logits are present because ``run_audit`` also reports the
    per-side latent geometry; they are not used by the hand-made terms of the analytic check.
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(5)
        self.z_node_mean = torch.randn(6, 4)
        self.z_edge_mean = torch.randn(5, 4)
        self.prediction_logits = torch.randn(6, 5)
        self.hgcn_circ = nn.Linear(3, 3, bias=False)        # similarity_view
        self.pair_predictor = nn.Linear(3, 1, bias=False)   # prediction_head
        self.residual_scale = nn.Parameter(torch.tensor([0.5]))  # untouched by any term
        self.secret = nn.Linear(3, 3, bias=False)           # lands in 'other'


def state_dicts_equal(left, right):
    if isinstance(left, dict):
        return set(left) == set(right) and all(state_dicts_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(state_dicts_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    return left == right


def check_group_mapping_covers_every_parameter():
    for head in ('residual_mlp', 'mlp'):
        model = RADVL(6, 5, model_arguments(head))
        groups = parameter_groups(model)
        counted = sum(len(groups[group]) for group in groups)
        trainable = sum(1 for parameter in model.parameters() if parameter.requires_grad)
        assert counted == trainable, (head, counted, trainable)
        assert groups['other'] == [], [name for name, _ in groups['other']]
        assert groups['similarity_view'] and groups['covae_encoder']
        assert groups['fusion'] and groups['prediction_head']
        assert bool(groups['residual_scale']) == (head == 'residual_mlp')


def check_subterms_carry_their_weights_and_sum_to_the_association():
    arguments = model_arguments()
    arguments.lambda_av, arguments.lambda_cov, arguments.lambda_align = 0.37, 4.0, 2.5
    torch.manual_seed(1234)
    model = RADVL(6, 5, arguments)
    _, parts = run_objective(model, radvl_objective_balanced, arguments)
    for name in ('weighted_reconstruction', 'weighted_kl', 'weighted_covariance_term'):
        assert name in parts, name
    total = (parts['weighted_reconstruction'] + parts['weighted_kl']
             + parts['weighted_covariance_term'])
    assert torch.allclose(total, parts['weighted_association_view'], rtol=1e-6, atol=1e-8)
    assert torch.allclose(parts['weighted_reconstruction'],
                          arguments.lambda_av * parts['reconstruction_bce'], rtol=1e-6, atol=1e-8)
    assert torch.allclose(parts['weighted_kl'],
                          arguments.lambda_av * parts['kl_standard'], rtol=1e-6, atol=1e-8)
    assert torch.allclose(parts['weighted_covariance_term'],
                          arguments.lambda_av * arguments.lambda_cov * parts['covariance_raw'],
                          rtol=1e-6, atol=1e-8)
    # The total objective is untouched by exposing the sub-terms.
    reference = (parts['prediction_bce'] + parts['weighted_association_view']
                 + parts['weighted_alignment'])
    assert torch.allclose(reference, parts['prediction_bce'] + parts['weighted_reconstruction']
                          + parts['weighted_kl'] + parts['weighted_covariance_term']
                          + parts['weighted_alignment'], rtol=1e-6, atol=1e-8)


def check_reported_norms_and_cosines_are_analytically_exact():
    torch.manual_seed(3)
    model = FakeAuditModel()
    x, y, z = torch.randn(3), torch.randn(3), torch.randn(3)
    shared_term = (model.hgcn_circ.weight * x).sum()      # gradient norm = sqrt(3) * ||x||
    head_term = (model.pair_predictor.weight * y).sum()   # gradient norm = ||y||
    other_term = (model.secret.weight * z).sum()
    parts = dict(prediction_bce=shared_term + head_term,
                 weighted_reconstruction=2.0 * shared_term,
                 weighted_kl=torch.zeros(()),                     # a term with no gradient at all
                 weighted_covariance_term=-3.0 * shared_term,
                 weighted_alignment=-1.0 * shared_term)
    audit = run_audit(model, parts, call_index=1)
    rows = {row['term']: row for row in audit['terms']}
    shared_norm = (3 ** 0.5) * float(x.norm())
    prediction_norm = (3 * float(x.norm()) ** 2 + float(y.norm()) ** 2) ** 0.5
    assert abs(rows['prediction']['grad_norm_all'] - prediction_norm) < 1e-5
    assert abs(rows['prediction']['grad_norm_prediction_head'] - float(y.norm())) < 1e-5
    assert abs(rows['reconstruction']['grad_norm_all'] - 2 * shared_norm) < 1e-5
    assert abs(rows['covariance']['grad_norm_all'] - 3 * shared_norm) < 1e-5
    assert rows['kl']['grad_norm_all'] == 0.0
    assert abs(rows['kl']['share_of_total_gradient']) < 1e-12
    # The aggregate is the sum of its parts: 2 - 3 + 0 = -1 times the shared gradient.
    assert abs(rows[AGGREGATE_LABEL]['grad_norm_all'] - shared_norm) < 1e-5
    assert rows[AGGREGATE_LABEL]['is_aggregate'] == 1
    assert math.isnan(rows[AGGREGATE_LABEL]['share_of_total_gradient'])
    assert rows['prediction']['is_aggregate'] == 0
    cosines = {(row['term_a'], row['term_b']): row for row in audit['cosines']}
    assert abs(cosines[('prediction', AGGREGATE_LABEL)]['cosine'] + 1.0) < 1e-6   # anti-parallel
    assert abs(cosines[('covariance', 'alignment')]['cosine'] - 1.0) < 1e-6       # parallel
    # Pairs are emitted in TERM_KEYS order, so the KL pair is (prediction, kl).
    assert math.isnan(cosines[('prediction', 'kl')]['cosine'])                    # no support
    assert cosines[('prediction', 'kl')]['common_parameter_count'] == 0
    assert audit['unassigned_parameter_count'] == 1
    assert audit['rng_state_unchanged_all'] is True
    assert audit['rng_state_unchanged'] == ''


def check_aggregate_equals_a_direct_gradient_of_the_association_term():
    arguments = model_arguments()
    torch.manual_seed(7)
    model = RADVL(6, 5, arguments)
    inputs = build_inputs()
    torch.manual_seed(99)
    _, reconstruction, _, _, covariance, alignment = model(*inputs)
    total, parts = radvl_objective_balanced(model, reconstruction, PAIRS['positive'],
                                                 PAIRS['negative'], covariance, alignment, arguments)
    audit = run_audit(model, parts, call_index=1)
    reported = {row['term']: row for row in audit['terms']}[AGGREGATE_LABEL]['grad_norm_all']
    parameters = [p for p in model.parameters() if p.requires_grad]
    direct = torch.autograd.grad(parts['weighted_association_view'], parameters,
                                 retain_graph=True, allow_unused=True)
    norm = sum(float(g.detach().pow(2).sum()) for g in direct if g is not None) ** 0.5
    assert abs(reported - norm) <= 1e-5 * max(1.0, norm), (reported, norm)
    # The sub-terms must have been audited on the same graph the loop will backpropagate.
    assert torch.isfinite(total)


def check_latent_geometry_matches_an_independent_reference():
    torch.manual_seed(11)
    mean = 1.5 * torch.randn(40, 12)
    report = latent_geometry(mean)
    values = mean.detach().numpy().astype(np.float64)
    covariance = np.cov(values, rowvar=False, ddof=1)
    off = ~np.eye(values.shape[1], dtype=bool)
    standard_deviation = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(standard_deviation, standard_deviation)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    assert abs(report['mean_abs_offdiagonal_covariance'] - np.abs(covariance[off]).mean()) < 1e-9
    assert abs(report['max_abs_offdiagonal_covariance'] - np.abs(covariance[off]).max()) < 1e-9
    assert abs(report['mean_abs_offdiagonal_correlation'] - np.abs(correlation[off]).mean()) < 1e-9
    assert abs(report['max_abs_offdiagonal_correlation'] - np.abs(correlation[off]).max()) < 1e-9
    assert abs(report['covariance_penalty_side']
               - (covariance[off] ** 2).sum() / (12 * 11)) < 1e-12
    assert abs(report['effective_rank']
               - (eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())) < 1e-9
    assert report['correlation_pair_count'] == 12 * 11
    assert report['usable_dimension_fraction'] == 1.0
    # The reported side penalty must reproduce the objective's covariance penalty (which is
    # computed in float32 on the torch tensors, hence the loose comparison).
    other = 0.5 * torch.randn(25, 12)
    combined = 0.5 * (latent_geometry(mean)['covariance_penalty_side']
                      + latent_geometry(other)['covariance_penalty_side'])
    expected = float(covariance_penalty(mean, other))
    assert abs(combined - expected) < 1e-6, (combined, expected)
    # A degenerate constant dimension must be excluded, not divided by zero.
    degenerate = torch.randn(30, 5).clone()
    degenerate[:, 2] = 3.0
    report = latent_geometry(degenerate)
    for key, value in report.items():
        assert not (isinstance(value, float) and math.isnan(value)), key
    assert report['usable_dimension_fraction'] == 4 / 5
    assert report['correlation_pair_count'] == 4 * 3
    assert report['latent_std_min'] == 0.0


def check_audit_is_observer_only():
    arguments = model_arguments()
    torch.manual_seed(1234)
    model = RADVL(6, 5, arguments)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    # Warm the optimiser up so that its state is non-trivial before the audit.
    total, _ = run_objective(model, radvl_objective_balanced, arguments)
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    optimizer.zero_grad()
    inputs = build_inputs()
    torch.manual_seed(99)
    _, reconstruction, _, _, covariance, alignment = model(*inputs)
    parameters_before = {name: parameter.detach().clone()
                         for name, parameter in model.named_parameters()}
    gradients_before = {name: (None if parameter.grad is None else parameter.grad.detach().clone())
                        for name, parameter in model.named_parameters()}
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    rng_before = torch.get_rng_state()
    objective = GradientAuditObjective(radvl_objective_balanced, every=1)
    audited_total, _ = objective(model, reconstruction, PAIRS['positive'], PAIRS['negative'],
                                 covariance, alignment, arguments)
    assert model.gradient_audit is not None
    assert model.gradient_audit['rng_state_unchanged_all'] is True
    assert model.gradient_audit['rng_state_unchanged'] == ''
    assert torch.equal(rng_before, torch.get_rng_state())
    assert state_dicts_equal(optimizer_before, optimizer.state_dict())
    for name, parameter in model.named_parameters():
        assert torch.equal(parameters_before[name], parameter.detach()), name
        # .grad must be exactly what it was before the audit.  Comparing instead of asserting
        # None keeps the check independent of torch's zero_grad(set_to_none=...) default.
        before, after = gradients_before[name], parameter.grad
        assert (before is None) == (after is None), name
        if before is not None:
            assert torch.equal(before, after), name
    assert torch.isfinite(audited_total)
    # The graph must survive the audit: the caller's backward still works and matches the
    # unaudited path exactly.
    audited_total.backward()
    audited_gradients = {name: parameter.grad
                         for name, parameter in model.named_parameters()}
    assert any(gradient is not None for gradient in audited_gradients.values())
    return audited_total


def check_audit_preserves_the_training_gradients():
    arguments = model_arguments()

    def gradients(use_audit):
        torch.manual_seed(1234)
        model = RADVL(6, 5, arguments)
        inputs = build_inputs()
        torch.manual_seed(99)
        _, reconstruction, _, _, covariance, alignment = model(*inputs)
        if use_audit:
            objective = GradientAuditObjective(radvl_objective_balanced, every=1)
            total, _ = objective(model, reconstruction, PAIRS['positive'], PAIRS['negative'],
                                 covariance, alignment, arguments)
            assert model.gradient_audit is not None
        else:
            total, _ = radvl_objective_balanced(model, reconstruction, PAIRS['positive'],
                                                     PAIRS['negative'], covariance, alignment,
                                                     arguments)
            assert getattr(model, 'gradient_audit', None) is None
        model.zero_grad()
        total.backward()
        return float(total), {name: parameter.grad for name, parameter in model.named_parameters()}

    total_plain, grads_plain = gradients(False)
    total_audited, grads_audited = gradients(True)
    assert total_plain == total_audited
    assert set(grads_plain) == set(grads_audited)
    for name, value in grads_plain.items():
        other = grads_audited[name]
        assert (value is None) == (other is None), name
        if value is not None:
            assert torch.equal(value, other), name


def check_audit_can_be_disabled():
    arguments = model_arguments()
    torch.manual_seed(1234)
    model = RADVL(6, 5, arguments)
    inputs = build_inputs()
    torch.manual_seed(99)
    _, reconstruction, _, _, covariance, alignment = model(*inputs)
    positive = torch.tensor([[0, 0], [1, 1], [2, 2]])
    negative = torch.tensor([[0, 1], [1, 0], [5, 2]])
    objective = GradientAuditObjective(radvl_objective_balanced, every=0)
    total, _ = objective(model, reconstruction, positive, negative, covariance, alignment, arguments)
    assert model.gradient_audit is None
    reference, _ = radvl_objective_balanced(model, reconstruction, positive, negative,
                                                 covariance, alignment, arguments)
    assert float(total) == float(reference)


if __name__ == '__main__':
    checks = [value for name, value in sorted(globals().items()) if name.startswith('check_')]
    for check in checks:
        check()
        print('OK', check.__name__)
    print(f'{len(checks)} checks passed')

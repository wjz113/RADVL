"""RADVL gradient diagnostics."""
import os

import numpy as np
import torch


TERM_KEYS = (('prediction', 'prediction_bce'),
             ('reconstruction', 'weighted_reconstruction'),
             ('kl', 'weighted_kl'),
             ('covariance', 'weighted_covariance_term'),
             ('alignment', 'weighted_alignment'))

AGGREGATE_LABEL = 'association'
AGGREGATE_SOURCES = ('reconstruction', 'kl', 'covariance')

# Top-level module names of the model, grouped the way the review asks for.  Every trainable
# parameter must land in one of these groups or in 'other', which is counted and reported so a
# renamed attribute cannot silently drop out of the audit.
PARAMETER_GROUPS = (
    ('similarity_view', ('circ_fastkan', 'dr_fastkan', 'hgcn_circ', 'hgcn_dr', 'circ_gin_encoder',
                         'dr_gin_encoder', 'circ_global_projection', 'dr_global_projection',
                         'circ_similarity_projection', 'dr_similarity_projection')),
    ('covae_encoder', ('node_encoders1', 'hyperedge_encoders1', 'circ_mean', 'circ_log_variance',
                       'drug_mean', 'drug_log_variance', 'circ_association_projection',
                       'dr_association_projection')),
    ('fusion', ('circ_fusion', 'dr_fusion')),
    ('prediction_head', ('pair_predictor',)),
    ('residual_scale', ('residual_scale',)),
)

GROUP_NAMES = tuple(name for name, _ in PARAMETER_GROUPS) + ('other',)
LATENT_GEOMETRY_SIDES = (('circ', 'z_node_mean'), ('drug', 'z_edge_mean'))


def audit_interval():
    """Epochs between audits, from ``GRADIENT_AUDIT_EVERY`` (default 5, 0 disables)."""
    raw = os.environ.get('GRADIENT_AUDIT_EVERY', '5')
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f'GRADIENT_AUDIT_EVERY must be an integer, got {raw!r}')
    if value < 0:
        raise ValueError(f'GRADIENT_AUDIT_EVERY must be nonnegative, got {value}')
    return value


def parameter_groups(model):
    """Map every trainable parameter to a named group, preserving a deterministic order."""
    prefix_to_group = {prefix: name for name, prefixes in PARAMETER_GROUPS for prefix in prefixes}
    groups = {name: [] for name in GROUP_NAMES}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        groups[prefix_to_group.get(name.split('.')[0], 'other')].append((name, parameter))
    return groups


def rng_fingerprint():
    """CPU state always; CUDA states only when CUDA is already initialised (never initialises it)."""
    states = [('cpu', torch.get_rng_state())]
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        states.extend(('cuda:%d' % index, state)
                      for index, state in enumerate(torch.cuda.get_rng_state_all()))
    return states


def fingerprints_equal(before, after):
    if len(before) != len(after):
        return False
    return all(name == other and torch.equal(state, other_state)
               for (name, state), (other, other_state) in zip(before, after))


def latent_geometry(mean, tolerance=1e-8):
    """Covariance, correlation and effective rank of one side's posterior means.

    Computed in float64 on the CPU from a detached copy, so it cannot touch the autograd graph,
    the parameters or any RNG.  Dimensions with a standard deviation below ``tolerance`` are
    excluded from the correlations instead of producing a division by zero.
    """
    values = mean.detach().to(torch.float64).cpu().numpy()
    node_count, dimension = values.shape
    if dimension < 2:
        raise ValueError('Latent geometry requires a latent width of at least 2')
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (node_count - 1)
    standard_deviation = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    scale = np.outer(standard_deviation, standard_deviation)
    usable = np.outer(standard_deviation > tolerance, standard_deviation > tolerance)
    correlation = np.zeros_like(covariance)
    correlation[usable] = covariance[usable] / scale[usable]
    off_diagonal = ~np.eye(dimension, dtype=bool)
    off_covariance = np.abs(covariance[off_diagonal])
    off_pairs = off_diagonal & usable
    off_correlation = np.abs(correlation[off_pairs]) if off_pairs.any() else np.zeros(0)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    squared = float((eigenvalues ** 2).sum())
    effective_rank = float(eigenvalues.sum() ** 2 / squared) if squared > 0 else 0.0
    return dict(
        node_count=node_count, dimension=dimension,
        correlation_pair_count=int(off_pairs.sum()),
        usable_dimension_fraction=float((standard_deviation > tolerance).mean()),
        mean_abs_offdiagonal_covariance=float(off_covariance.mean()),
        max_abs_offdiagonal_covariance=float(off_covariance.max()),
        mean_abs_offdiagonal_correlation=float(off_correlation.mean()) if off_correlation.size else 0.0,
        max_abs_offdiagonal_correlation=float(off_correlation.max()) if off_correlation.size else 0.0,
        covariance_penalty_side=float((off_covariance ** 2).sum() / (dimension * (dimension - 1))),
        effective_rank=effective_rank,
        effective_rank_fraction=effective_rank / dimension,
        latent_std_mean=float(standard_deviation.mean()),
        latent_std_min=float(standard_deviation.min()),
        latent_std_max=float(standard_deviation.max()))


def _norm(indices, gradients):
    total = 0.0
    for index in indices:
        gradient = gradients[index]
        if gradient is not None:
            total += float(gradient.detach().pow(2).sum())
    return total ** 0.5


def _add_gradients(sources, index):
    total = None
    for gradients in sources:
        gradient = gradients[index]
        if gradient is None:
            continue
        total = gradient if total is None else total + gradient
    return total


def run_audit(model, parts, call_index):
    """Per-term gradient norms, shares, cosines, latent geometry and the observer-only check."""
    rng_before = rng_fingerprint()
    groups = parameter_groups(model)
    parameters, owners = [], []
    for group in GROUP_NAMES:
        for _, parameter in groups[group]:
            parameters.append(parameter)
            owners.append(group)
    if not parameters:
        raise RuntimeError('The model has no trainable parameters to audit')
    independent = []
    for label, key in TERM_KEYS:
        if key not in parts:
            raise KeyError(f'Objective parts lack {key!r}, cannot audit the {label!r} term')
        independent.append((label, parts[key]))
    labels = [label for label, _ in independent]
    term_gradients = {}
    for label, value in independent:
        if not value.requires_grad:
            term_gradients[label] = [None] * len(parameters)
            continue
        term_gradients[label] = list(torch.autograd.grad(value, parameters, retain_graph=True,
                                                         allow_unused=True))
    # The aggregate is the sum of its parts, so its gradient is the sum of their gradients:
    # exact by linearity, and it saves a fifth traversal.
    aggregate = [_add_gradients([term_gradients[source] for source in AGGREGATE_SOURCES], index)
                 for index in range(len(parameters))]
    term_gradients[AGGREGATE_LABEL] = aggregate
    # Support and shares come from the independent terms only; the aggregate would inflate both.
    support = [sum(gradients[index] is not None for gradients in
                   (term_gradients[label] for label in labels)) for index in range(len(parameters))]
    shared = [index for index, count in enumerate(support) if count >= 2]
    all_indices = list(range(len(parameters)))
    group_indices = {group: [index for index, owner in enumerate(owners) if owner == group]
                     for group in GROUP_NAMES}
    prediction_norm = _norm(all_indices, term_gradients['prediction'])
    combined_norm = sum(_norm(all_indices, term_gradients[label]) ** 2 for label in labels) ** 0.5
    rows = []
    for label in labels + [AGGREGATE_LABEL]:
        gradients = term_gradients[label]
        norm = _norm(all_indices, gradients)
        row = dict(term=label, is_aggregate=int(label == AGGREGATE_LABEL),
                   grad_norm_all=norm, grad_norm_shared=_norm(shared, gradients),
                   ratio_to_prediction=(norm / prediction_norm) if prediction_norm > 0 else float('nan'),
                   share_of_total_gradient=(float('nan') if label == AGGREGATE_LABEL
                                            else (norm / combined_norm) if combined_norm > 0 else float('nan')),
                   shared_parameter_count=len(shared))
        for group in GROUP_NAMES:
            row['grad_norm_' + group] = _norm(group_indices[group], gradients)
        rows.append(row)
    cosines = []
    pair_labels = labels + [AGGREGATE_LABEL]
    for first in range(len(pair_labels)):
        for second in range(first + 1, len(pair_labels)):
            left = term_gradients[pair_labels[first]]
            right = term_gradients[pair_labels[second]]
            # Cosine on the parameters the two terms both touch; where they share none it is
            # undefined, so it is reported as NaN together with common_parameter_count = 0.
            common = [index for index in range(len(parameters))
                      if left[index] is not None and right[index] is not None]
            dot = left_norm = right_norm = 0.0
            for index in common:
                dot += float((left[index].detach() * right[index].detach()).sum())
                left_norm += float(left[index].detach().pow(2).sum())
                right_norm += float(right[index].detach().pow(2).sum())
            cosines.append(dict(term_a=pair_labels[first], term_b=pair_labels[second],
                                cosine=(dot / ((left_norm ** 0.5) * (right_norm ** 0.5)))
                                if left_norm > 0 and right_norm > 0 else float('nan'),
                                common_parameter_count=len(common),
                                shared_parameter_count=len(shared)))
    geometry = {}
    for side, attribute in LATENT_GEOMETRY_SIDES:
        latent = getattr(model, attribute, None)
        if latent is None:
            raise AttributeError(f'The model does not expose {attribute!r} for the geometry report')
        geometry[side] = latent_geometry(latent)
    from_latents = 0.5 * (geometry['circ']['covariance_penalty_side']
                          + geometry['drug']['covariance_penalty_side'])
    objective_penalty = float(parts['covariance_raw']) if 'covariance_raw' in parts else float('nan')
    for side in geometry:
        geometry[side]['covariance_penalty_from_latents'] = from_latents
        geometry[side]['covariance_penalty_objective'] = objective_penalty
        geometry[side]['covariance_penalty_absolute_difference'] = abs(from_latents - objective_penalty)
        geometry[side]['latent_logit_abs_mean'] = float(model.prediction_logits.detach().abs().mean())
    rng_after = rng_fingerprint()
    return dict(call_index=call_index, terms=rows, cosines=cosines, geometry=geometry,
                shared_parameter_count=len(shared), parameter_count=len(parameters),
                unassigned_parameter_count=len(group_indices['other']),
                rng_state_unchanged=rng_fingerprint_labels(rng_before, rng_after),
                rng_state_unchanged_all=fingerprints_equal(rng_before, rng_after))


def rng_fingerprint_labels(before, after):
    """Comma-separated names whose RNG state changed; empty string means all states were stable."""
    changed = [name for (name, state), (other, other_state) in zip(before, after)
               if name != other or not torch.equal(state, other_state)]
    if len(before) != len(after):
        changed.append('device_count')
    return ','.join(changed)


class GradientAuditObjective:
    """Wraps a manuscript objective, audits the gradients and returns its result unchanged."""

    def __init__(self, base_objective, every=None):
        self.base_objective = base_objective
        self.every = audit_interval() if every is None else int(every)
        if self.every < 0:
            raise ValueError('every must be nonnegative')
        self.call_index = 0
        self.model = None

    def __call__(self, model, reconstruction_logits, positive_pairs, negative_pairs, covariance,
                 alignment, args):
        total, parts = self.base_objective(model, reconstruction_logits, positive_pairs,
                                           negative_pairs, covariance, alignment, args)
        if model is not self.model:
            # A new fold builds a new model; keeping the reference prevents id reuse.
            self.model, self.call_index = model, 0
        self.call_index += 1
        model.gradient_audit = None
        if self.every and self.call_index % self.every == 0:
            model.gradient_audit = run_audit(model, parts, self.call_index)
        return total, parts

"""RADVL objectives pair scaling."""
import torch
import torch.nn.functional as F

from radvl.objectives import covariance_penalty, mean_node_kl

__all__ = ['covariance_penalty', 'mean_node_kl', 'pair_scale',
           'radvl_objective_pairscaled']


def pair_scale(args, pair_count):
    """Multiplier applied to lambda_1 and lambda_2.

    ``pairs``  -> |Omega|, the number of training pairs (summed-loss scale)
    ``unity``  -> 1.0, reproducing the unscaled objective exactly
    """
    mode = getattr(args, 'lambda_scale', 'pairs')
    if mode == 'pairs':
        if pair_count < 1:
            raise ValueError('Pair-scaled objective requires at least one training pair')
        return float(pair_count)
    if mode == 'unity':
        return 1.0
    raise ValueError(f'Unknown lambda scaling mode: {mode}')


def radvl_objective_pairscaled(model, reconstruction_logits, positive_pairs, negative_pairs,
                                    covariance, alignment, args):
    """Eq. (45) with |Omega|-scaled auxiliary weights; every other term is unchanged.

    The call signature is identical to ``radvl_objective`` so the shared training
    loop can use either objective without any other change.
    """
    if not len(positive_pairs) or not len(negative_pairs):
        raise ValueError('Both training-positive and training-negative sets must be nonempty')
    pair_count = int(len(positive_pairs) + len(negative_pairs))
    scale = pair_scale(args, pair_count)
    pos = (positive_pairs[:, 0], positive_pairs[:, 1])
    neg = (negative_pairs[:, 0], negative_pairs[:, 1])
    prediction_pos, prediction_neg = model.prediction_logits[pos], model.prediction_logits[neg]
    prediction = (
        F.binary_cross_entropy_with_logits(prediction_pos, torch.ones_like(prediction_pos), reduction='sum')
        + F.binary_cross_entropy_with_logits(prediction_neg, torch.zeros_like(prediction_neg), reduction='sum'))
    reconstruction_pos, reconstruction_neg = reconstruction_logits[pos], reconstruction_logits[neg]
    positive_bce = F.binary_cross_entropy_with_logits(
        reconstruction_pos, torch.ones_like(reconstruction_pos), reduction='mean')
    negative_bce = F.binary_cross_entropy_with_logits(
        reconstruction_neg, torch.zeros_like(reconstruction_neg), reduction='mean')
    reconstruction = args.eta_pos * positive_bce + args.eta_neg * negative_bce
    kl_circ = mean_node_kl(model.z_node_mean, model.z_node_log_var)
    kl_drug = mean_node_kl(model.z_edge_mean, model.z_edge_log_var)
    kl = kl_circ + kl_drug
    weighted_covariance = args.lambda_cov * covariance
    association_view = reconstruction + kl + weighted_covariance
    weighted_association = scale * args.lambda_av * association_view
    weighted_alignment = scale * args.lambda_align * alignment
    total = prediction + weighted_association + weighted_alignment
    auxiliary = weighted_association + weighted_alignment
    parts = dict(
        prediction_bce=prediction, reconstruction_positive_bce=positive_bce,
        reconstruction_negative_bce=negative_bce, reconstruction_bce=reconstruction,
        kl_circ=kl_circ, kl_drug=kl_drug, kl_standard=kl,
        covariance_raw=covariance, weighted_covariance=weighted_covariance,
        association_view=association_view, weighted_association_view=weighted_association,
        alignment_raw=alignment, weighted_alignment=weighted_alignment,
        # Objective-balance evidence (all tensors: the shared recorder calls .detach().item()).
        pair_count=torch.as_tensor(float(pair_count), device=total.device),
        lambda_scale=torch.as_tensor(scale, device=total.device),
        lambda_av_effective=torch.as_tensor(scale * args.lambda_av, device=total.device),
        lambda_align_effective=torch.as_tensor(scale * args.lambda_align, device=total.device),
        prediction_share_of_total=prediction / total,
        auxiliary_share_of_total=auxiliary / total)
    if getattr(args, 'log_normalized_objective', 'on') == 'on':
        # Pure logging: dividing every term by |Omega| gives the per-pair view of the same
        # objective, so the recorded loss is directly comparable across runs.  Nothing here
        # enters ``total`` and therefore nothing here affects the gradients.  The three
        # parts sum exactly to normalized_total_loss.
        parts.update(
            normalized_total_loss=total / pair_count,
            normalized_prediction_bce=prediction / pair_count,
            normalized_weighted_association=weighted_association / pair_count,
            normalized_weighted_alignment=weighted_alignment / pair_count)
    return total, parts

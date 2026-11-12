"""RADVL objectives balanced."""
import torch
import torch.nn.functional as F

from radvl.objectives import covariance_penalty, mean_node_kl

__all__ = ['covariance_penalty', 'mean_node_kl', 'radvl_objective_balanced']


def radvl_objective_balanced(model, reconstruction_logits, positive_pairs, negative_pairs,
                                   covariance, alignment, args):
    """Eq. (45) with the prediction term averaged over the training pairs."""
    if not len(positive_pairs) or not len(negative_pairs):
        raise ValueError('Both training-positive and training-negative sets must be nonempty')
    pair_count = int(len(positive_pairs) + len(negative_pairs))
    pos = (positive_pairs[:, 0], positive_pairs[:, 1])
    neg = (negative_pairs[:, 0], negative_pairs[:, 1])
    prediction_pos, prediction_neg = model.prediction_logits[pos], model.prediction_logits[neg]
    prediction_sum = (
        F.binary_cross_entropy_with_logits(prediction_pos, torch.ones_like(prediction_pos), reduction='sum')
        + F.binary_cross_entropy_with_logits(prediction_neg, torch.zeros_like(prediction_neg), reduction='sum'))
    prediction = prediction_sum / pair_count
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
    weighted_association = args.lambda_av * association_view
    weighted_alignment = args.lambda_align * alignment
    total = prediction + weighted_association + weighted_alignment
    auxiliary = weighted_association + weighted_alignment
    # The same three addends as weighted_association, exposed separately with their actual
    # weights (lambda_av, and lambda_av * lambda_cov for the covariance penalty) so the gradient
    # audit can attribute its contribution without another forward pass.  The total is unchanged:
    # weighted_reconstruction + weighted_kl + weighted_covariance_term == weighted_association.
    weighted_reconstruction = args.lambda_av * reconstruction
    weighted_kl = args.lambda_av * kl
    weighted_covariance_term = args.lambda_av * weighted_covariance
    return total, dict(
        prediction_bce=prediction, prediction_bce_sum=prediction_sum,
        reconstruction_positive_bce=positive_bce, reconstruction_negative_bce=negative_bce,
        reconstruction_bce=reconstruction, kl_circ=kl_circ, kl_drug=kl_drug, kl_standard=kl,
        covariance_raw=covariance, weighted_covariance=weighted_covariance,
        weighted_reconstruction=weighted_reconstruction, weighted_kl=weighted_kl,
        weighted_covariance_term=weighted_covariance_term,
        association_view=association_view, weighted_association_view=weighted_association,
        alignment_raw=alignment, weighted_alignment=weighted_alignment,
        pair_count=torch.as_tensor(float(pair_count), device=total.device),
        prediction_share_of_total=prediction / total,
        auxiliary_share_of_total=auxiliary / total)

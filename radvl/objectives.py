"""RADVL objectives."""
import torch
import torch.nn.functional as F


def covariance_penalty(circ_mu, drug_mu):
    dimension = circ_mu.shape[1]
    if dimension < 2 or drug_mu.shape[1] != dimension:
        raise ValueError('Covariance requires equal latent widths >= 2')
    penalties = []
    for mean in (circ_mu, drug_mu):
        if len(mean) < 2:
            raise ValueError('Sample covariance requires at least two entities per side')
        centered = mean - mean.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / (len(mean) - 1)
        off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
        penalties.append(off_diagonal.square().sum())
    return (penalties[0] + penalties[1]) / (2 * dimension * (dimension - 1))


def mean_node_kl(mean, log_variance):
    return 0.5 * (mean.square() + log_variance.exp() - 1 - log_variance).sum(dim=1).mean()


def radvl_objective(model, reconstruction_logits, positive_pairs, negative_pairs,
                         covariance, alignment, args):
    if not len(positive_pairs) or not len(negative_pairs):
        raise ValueError('Both training-positive and training-negative sets must be nonempty')
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
    weighted_association = args.lambda_av * association_view
    weighted_alignment = args.lambda_align * alignment
    total = prediction + weighted_association + weighted_alignment
    return total, dict(prediction_bce=prediction, reconstruction_positive_bce=positive_bce,
                      reconstruction_negative_bce=negative_bce, reconstruction_bce=reconstruction,
                      kl_circ=kl_circ, kl_drug=kl_drug, kl_standard=kl,
                      covariance_raw=covariance, weighted_covariance=weighted_covariance,
                      association_view=association_view, weighted_association_view=weighted_association,
                      alignment_raw=alignment, weighted_alignment=weighted_alignment)

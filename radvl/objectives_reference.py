"""RADVL objectives reference."""
import torch
import torch.nn.functional as F


def covariance_penalty(circ_mu, drug_mu):
    """Eq. (30): both within-view covariances, squared off-diagonals only."""
    dimension = circ_mu.shape[1]
    if dimension != drug_mu.shape[1] or dimension < 2:
        raise ValueError('Both latent dimensions must match and be >= 2')
    penalties = []
    for mean in (circ_mu, drug_mu):
        if mean.shape[0] < 2:
            raise ValueError('Covariance requires at least two entities per side')
        centered = mean - mean.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / (mean.shape[0] - 1)
        off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
        penalties.append(off_diagonal.square().sum())
    return sum(penalties) / (2 * dimension * (dimension - 1))


def mean_node_kl(mean, log_variance):
    """KL to N(0,I), summed over dimensions and averaged over nodes."""
    return 0.5 * (mean.square() + log_variance.exp()
                  - 1 - log_variance).sum(dim=1).mean()


def paper_objective(score, reconstruction_logits, model, positive_pairs,
                    negative_pairs, covariance, alignment, args):
    positive_pairs = positive_pairs.to(score.device)
    negative_pairs = negative_pairs.to(score.device)
    pos = (positive_pairs[:, 0], positive_pairs[:, 1])
    neg = (negative_pairs[:, 0], negative_pairs[:, 1])
    positive_scores, negative_scores = score[pos], score[neg]
    # Eq. (44) is a sum over the selected training pairs.
    prediction = (
        F.binary_cross_entropy(positive_scores, torch.ones_like(positive_scores),
                               reduction='sum')
        + F.binary_cross_entropy(negative_scores, torch.zeros_like(negative_scores),
                                 reduction='sum')
    )
    positive_logits = reconstruction_logits[pos]
    negative_logits = reconstruction_logits[neg]
    # Eq. (27): separately normalized positive/negative reconstruction terms.
    reconstruction = (
        args.eta_pos * F.binary_cross_entropy_with_logits(
            positive_logits, torch.ones_like(positive_logits), reduction='mean')
        + args.eta_neg * F.binary_cross_entropy_with_logits(
            negative_logits, torch.zeros_like(negative_logits), reduction='mean')
    )
    kl_circ = mean_node_kl(model.z_node_mean, model.z_node_log_var)
    kl_drug = mean_node_kl(model.z_edge_mean, model.z_edge_log_var)
    kl = kl_circ + kl_drug
    association_view = reconstruction + kl + args.co_lambda * covariance
    total = prediction + args.lambda_av * association_view + args.lambda_align * alignment
    return total, dict(prediction=prediction, reconstruction=reconstruction,
                      kl=kl, kl_circ=kl_circ, kl_drug=kl_drug,
                      covariance=covariance, alignment=alignment,
                      association_view=association_view)

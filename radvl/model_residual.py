"""RADVL model residual."""
import torch
from torch import nn

from radvl.model_similarity import RADVL as WeightedRADVL


PREDICTION_HEADS = ('residual_mlp', 'mlp')


class RADVL(WeightedRADVL):
    def __init__(self, n_circ, n_drug, args):
        super().__init__(n_circ, n_drug, args)
        head = getattr(args, 'prediction_head', 'residual_mlp')
        if head not in PREDICTION_HEADS:
            raise ValueError(f'Unknown prediction head: {head}')
        self.prediction_head = head
        self.fusion_logits = None
        self.residual_gamma = None
        if head == 'residual_mlp':
            # gamma = tanh(a).  With a = 0 (the default) the fused branch starts switched off:
            # dL/d s_fusion = gamma = 0, so the head is trained only through dL/da, and the
            # measured result was that gamma stayed at 0.003-0.017 and the fusion branch
            # collapsed to AUC 0.5 in 3 of 5 folds.  --residual_init 1.0 (gamma = 0.76) trains
            # the fusion branch from the first step, which is the fair test of whether it
            # carries usable signal.
            self.residual_scale = nn.Parameter(torch.full((1,), float(getattr(args, 'residual_init', 0.0))))

    def association_logits(self):
        """The Eq. (26) inner-product decoder, used as the residual base score."""
        return self.z_node_sample @ self.z_edge_sample.T

    def pair_logits(self, circ, drug):
        fusion_logits = super().pair_logits(circ, drug)
        self.fusion_logits = fusion_logits
        if self.prediction_head != 'residual_mlp':
            return fusion_logits
        gamma = torch.tanh(self.residual_scale)
        self.residual_gamma = gamma.detach()
        return self.association_logits() + gamma * fusion_logits

    def forward(self, *forward_args, **forward_kwargs):
        result = super().forward(*forward_args, **forward_kwargs)
        if self.prediction_head == 'residual_mlp':
            # The four-way score comparison proposed for the residual ablation, as detached
            # branches: the association score, the fusion score, their unweighted sum (a
            # diagnostic control only) and the learned combination (already 'combined').
            with torch.no_grad():
                self.diagnostic_scores['fusion_only'] = torch.sigmoid(self.fusion_logits).detach()
                self.diagnostic_scores['association_plus_fusion'] = torch.sigmoid(
                    self.association_logits() + self.fusion_logits).detach()
        return result

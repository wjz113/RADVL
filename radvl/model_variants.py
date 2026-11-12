"""RADVL model variants."""
import torch

from radvl.model_balanced import RADVL as BalancedRADVL


class RADVL(BalancedRADVL):
    """The **balanced** model (the one that produced the 0.8901 baseline) plus two switches.

    An earlier revision subclassed ``model_residual`` instead.  That
    silently dropped the two mechanisms the balanced entry installs -- the dense deterministic
    neighbour aggregation (``A @ X`` instead of ``index_add_``) and the detached alignment weight
    -- and the resulting runs took the epoch-2 *collapsed* branch in 5/5 folds (alignment value
    about 720 instead of about 105, effective rank 1.2-1.6 instead of 4.1, test AUC 0.8786
    instead of 0.8901).  Subclassing the balanced model keeps the control entry on the same
    numerical path as the audit entry, which is what makes its results comparable.
    """
    def __init__(self, n_circ, n_drug, args):
        super().__init__(n_circ, n_drug, args)
        self.association_gradient = getattr(args, 'association_gradient', 'on')
        if self.association_gradient not in ('on', 'off'):
            raise ValueError(f'Unknown association_gradient: {self.association_gradient}')
        self.similarity_diagnostic = getattr(args, 'similarity_diagnostic', 'on')
        if self.similarity_diagnostic not in ('on', 'off'):
            raise ValueError(f'Unknown similarity_diagnostic: {self.similarity_diagnostic}')
        self.circ_similarity_view = None
        self.drug_similarity_view = None
        if self.similarity_diagnostic == 'on':
            self.circ_similarity_projection.register_forward_hook(self._capture_similarity('circ'))
            self.dr_similarity_projection.register_forward_hook(self._capture_similarity('drug'))

    def _capture_similarity(self, side):
        """Hook factory: stash the similarity-view embedding without recomputing the encoders."""
        attribute = side + '_similarity_view'

        def capture(module, inputs, output):
            setattr(self, attribute, output.detach())

        return capture

    def association_logits(self):
        """Eq. (26) inner-product decoder, optionally with its gradient path cut."""
        logits = super().association_logits()
        if self.association_gradient == 'off':
            return logits.detach()
        return logits

    def forward(self, *forward_args, **forward_kwargs):
        result = super().forward(*forward_args, **forward_kwargs)
        if self.similarity_diagnostic == 'on' and self.circ_similarity_view is not None:
            with torch.no_grad():
                self.diagnostic_scores['similarity_inner_product'] = torch.sigmoid(
                    self.circ_similarity_view @ self.drug_similarity_view.T).detach()
        return result

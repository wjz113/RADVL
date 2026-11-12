"""RADVL objectives alignment."""
import copy


class ScheduledAlignmentObjective:
    """Wraps an objective and scales ``lambda_align`` over the first ``--lambda_align_warmup`` epochs."""

    def __init__(self, base_objective):
        self.base_objective = base_objective
        self.call_index = 0
        self.model = None

    def reset(self):
        self.call_index = 0
        self.model = None

    def __call__(self, model, reconstruction_logits, positive_pairs, negative_pairs, covariance,
                 alignment, args):
        if model is not self.model:
            # A new fold builds a new model; keeping the reference prevents id reuse.
            self.model, self.call_index = model, 0
        self.call_index += 1
        warmup = int(getattr(args, 'lambda_align_warmup', 0) or 0)
        factor = 1.0 if warmup <= 0 else min(1.0, self.call_index / float(warmup))
        scheduled = copy.copy(args)
        scheduled.lambda_align = args.lambda_align * factor
        if factor != 1.0:
            # Keep the compatibility alias the recorders read consistent with the scheduled value.
            if hasattr(scheduled, 'fusion_alignment_weight'):
                scheduled.fusion_alignment_weight = scheduled.lambda_align
        total, parts = self.base_objective(model, reconstruction_logits, positive_pairs,
                                           negative_pairs, covariance, alignment, scheduled)
        model.alignment_schedule_factor = factor
        return total, parts

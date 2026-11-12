"""RADVL training variants."""
from radvl import training_core as radvl_base
from radvl import training_audit as audit
from radvl.settings_variants import parameter_parser
from radvl.model_variants import RADVL
from radvl.recorder_variants import RADVLResidualControlsRecorder
from radvl.objectives_variants import radvl_objective_controls
from radvl.objectives_alignment import ScheduledAlignmentObjective
from radvl.gradient_diagnostics import GradientAuditObjective, audit_interval


_WEIGHTED = audit.balanced.base
_WEIGHTED.RADVL = RADVL
_WEIGHTED.RADVLWeightedRecorder = RADVLResidualControlsRecorder
_WEIGHTED.RESULTS_ROOT = 'Results_radvl_aligned_residual_controls'

_ORIGINAL_OBJECTIVE = radvl_base.radvl_objective
# The schedule sits inside the audit, so the audit reports the scheduled alignment term.
_CONTROLS_OBJECTIVE = GradientAuditObjective(
    ScheduledAlignmentObjective(radvl_objective_controls))

RESULTS_ROOT = _WEIGHTED.RESULTS_ROOT


def main(args):
    print(f'Controls entry: loss_coefficients={args.loss_coefficients} '
          f'(lambda_rec={args.lambda_rec}, beta_kl={args.beta_kl}, lambda_cov_star={args.lambda_cov_star}); '
          f'association_gradient={args.association_gradient}; similarity_diagnostic={args.similarity_diagnostic}; '
          f'lambda_align={args.lambda_align} with warm-up={args.lambda_align_warmup} epoch(s); '
          f'prediction_head={args.prediction_head}; residual_init={args.residual_init}; '
          f'gradient audit every {audit_interval()} epoch(s) (GRADIENT_AUDIT_EVERY=0 disables it).', flush=True)
    radvl_base.radvl_objective = _CONTROLS_OBJECTIVE
    try:
        return _WEIGHTED.main(args)
    finally:
        radvl_base.radvl_objective = _ORIGINAL_OBJECTIVE


if __name__ == '__main__':
    main(parameter_parser())

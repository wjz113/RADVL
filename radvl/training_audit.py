"""RADVL training audit."""
from radvl import training_core as radvl_base
from radvl import training_balanced as balanced
from radvl.settings_residual import parameter_parser
from radvl.gradient_diagnostics import GradientAuditObjective, audit_interval
from radvl.recorder_diagnostics import RADVLResidualAuditRecorder


_WEIGHTED = balanced.base
_WEIGHTED.RADVLWeightedRecorder = RADVLResidualAuditRecorder
_WEIGHTED.RESULTS_ROOT = 'Results_radvl_aligned_residual_balanced_audit'

_ORIGINAL_OBJECTIVE = radvl_base.radvl_objective
_AUDITED_OBJECTIVE = GradientAuditObjective(balanced.radvl_objective_balanced)

RESULTS_ROOT = _WEIGHTED.RESULTS_ROOT


def main(args):
    print(f'Gradient audit: every {audit_interval()} epoch(s) (GRADIENT_AUDIT_EVERY=0 disables it); '
          f'audited terms = prediction, reconstruction, KL, covariance, alignment plus the '
          f'association aggregate, each already multiplied by its actual weight; per-side latent '
          f'covariance, correlation and effective rank; prediction loss = mean over |Omega| pairs; '
          f'prediction head = {args.prediction_head}. The audit is observer-only: parameters, '
          f'optimiser state, .grad and the RNG state are untouched.', flush=True)
    radvl_base.radvl_objective = _AUDITED_OBJECTIVE
    try:
        return _WEIGHTED.main(args)
    finally:
        radvl_base.radvl_objective = _ORIGINAL_OBJECTIVE


if __name__ == '__main__':
    main(parameter_parser())

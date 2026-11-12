"""RADVL training balanced."""
from radvl import training_core as radvl_base
from radvl import training_weighted as base
from radvl.settings_residual import parameter_parser
from radvl.model_balanced import RADVL
from radvl.recorder_balanced import RADVLResidualBalancedRecorder
from radvl.objectives_balanced import radvl_objective_balanced


base.RADVL = RADVL
base.RADVLWeightedRecorder = RADVLResidualBalancedRecorder
base.RESULTS_ROOT = 'Results_radvl_aligned_residual_balanced'
# The shared training loop resolves radvl_objective from its own module namespace.
_ORIGINAL_OBJECTIVE = radvl_base.radvl_objective

RESULTS_ROOT = base.RESULTS_ROOT


def main(args):
    print(f'Balanced entry: prediction loss = mean over |Omega| pairs (review section 4); '
          f'dense deterministic aggregation and detached alignment weight installed '
          f'(review sections 8.1, 2.3); prediction head = {args.prediction_head}.', flush=True)
    radvl_base.radvl_objective = radvl_objective_balanced
    try:
        return base.main(args)
    finally:
        radvl_base.radvl_objective = _ORIGINAL_OBJECTIVE


if __name__ == '__main__':
    main(parameter_parser())

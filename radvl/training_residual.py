"""RADVL training residual."""
from radvl import training_weighted as base
from radvl.settings_residual import parameter_parser
from radvl.model_residual import RADVL
from radvl.recorder_residual import RADVLResidualRecorder


# The three substitutions; everything else in the delegated main stays the A3 code path.
base.RADVL = RADVL
base.RADVLWeightedRecorder = RADVLResidualRecorder
base.RESULTS_ROOT = 'Results_radvl_aligned_residual'

RESULTS_ROOT = base.RESULTS_ROOT


def main(args):
    print(f'Residual prediction head: mode={args.prediction_head}; '
          f'logit = s_AV + tanh(a) * s_fusion with a initialised to zero; '
          f'gamma is logged per epoch to residual_scale.csv.', flush=True)
    return base.main(args)


if __name__ == '__main__':
    main(parameter_parser())

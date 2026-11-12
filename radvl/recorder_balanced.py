"""RADVL recorder balanced."""
import json
from pathlib import Path
import shutil

from radvl.recorder_residual import RADVLResidualRecorder
from radvl.data_protocol import file_hash


BALANCED_SOURCE_FILES = ('objectives_balanced.py',
                         'model_balanced.py',
                         'recorder_balanced.py',
                         'training_balanced.py',
                         'test_balanced_objective.py')


class RADVLResidualBalancedRecorder(RADVLResidualRecorder):
    def __init__(self, directory, args, k, dataset, protocol, requested_args):
        super().__init__(directory, args, k, dataset, protocol, requested_args)
        root = Path(__file__).resolve().parent
        config = json.loads((self.directory / 'run_config.json').read_text(encoding='utf-8'))
        for name in BALANCED_SOURCE_FILES:
            shutil.copy2(root / name, self.directory / 'source' / name)
            config['input_and_source_sha256']['source/' + name] = file_hash(root / name)
        config.update(
            model='radvl_aligned_covariance_CoVAE_weighted_graphs_residual_prediction_balanced',
            prediction_loss_reduction='Eq44 divided by |Omega| (mean form, review section 4)',
            loss_scale_policy=dict(
                form='L = mean_over_pairs(BCE) + lambda_av * L_AV + lambda_align * L_align',
                unchanged='every formula and every weight keeps its manuscript meaning; only the reduction changed',
                lambda_interpretation='with this reduction the auxiliary terms are no longer 3-4 orders of magnitude '
                                      'smaller: from the A3 logs lambda_av=0.01 gives about 21% of the total and '
                                      'lambda_align=0.02 gives about 55%, so the alignment term becomes dominant. A '
                                      '10-20% auxiliary share corresponds to lambda_av about 0.002-0.005 and '
                                      'lambda_align about 0.002-0.004',
                equivalence='algebraically equal to the _pairscaled objective divided by |Omega|, but not numerically '
                            'identical under AdamW because weight decay does not scale with the loss'),
            review_mechanisms=[
                'section 8.1: dense A_kNN @ X neighbour aggregation instead of index_add_, removing the CUDA atomics '
                'that made two identical runs diverge from epoch 2 (no parameter is touched, so the initialisation '
                'stays bit-identical to A3)',
                'section 2.3: the alignment weight omega enters the loss detached, removing the incentive to inflate '
                'the discrepancy D; the weight values and every diagnostic are unchanged',
                'section 3 (inherited from the residual entry): logit = s_AV + tanh(a) * s_fusion with a = 0 at '
                'initialisation'],
            not_implemented=['section 5 overlapping hyperedge membership: measured evidence says removing the '
                             'hard-partition degeneracy is worth about +0.0005 test AUC (kNN hypergraph 0.875634 vs '
                             'K-means 0.875164), i.e. inside the 0.0018 noise floor',
                             'section 6 attention diagnostics: the attention matrices are not logged, so the row-entropy '
                             'diagnostic needs a new logging field first',
                             'section 4 lambda_align warm-up: it needs the epoch inside the objective, which the shared '
                             'training loop does not pass'],
            evidence_notes=[
                'Run-to-run noise floor: at least 0.0018 mean test AUC for two identical computations, so every '
                'comparison must be paired across folds and seeds.',
                'The dense aggregation is mathematically equivalent but not bit-identical to the index_add_ version '
                '(different summation order), so small numerical differences are expected and are not evidence of a '
                'behaviour change.'],
            limitations=list(config.get('limitations', []))
                        + ['Three changes plus the inherited residual head and adaptive temperature; attribute the '
                           'result with the dedicated control entries rather than from this run alone.',
                           'This variant has not been trained by the editing task that created it.'])
        config['notes'].append('Mean-form prediction loss, dense deterministic aggregation and the detached alignment '
                               'weight are recorded in run_config.json.')
        self.write_json('run_config.json', config)

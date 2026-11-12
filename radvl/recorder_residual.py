"""RADVL recorder residual."""
import json
from pathlib import Path
import shutil

import torch

from radvl.recorder_similarity import RADVLWeightedRecorder
from radvl.recorder_base import append_csv
from radvl.data_protocol import file_hash


RESIDUAL_SOURCE_FILES = ('model_residual.py',
                         'settings_residual.py',
                         'recorder_residual.py',
                         'training_residual.py',
                         'test_residual_prediction.py')


class RADVLResidualRecorder(RADVLWeightedRecorder):
    def __init__(self, directory, args, k, dataset, protocol, requested_args):
        super().__init__(directory, args, k, dataset, protocol, requested_args)
        root = Path(__file__).resolve().parent
        config = json.loads((self.directory / 'run_config.json').read_text(encoding='utf-8'))
        for name in RESIDUAL_SOURCE_FILES:
            shutil.copy2(root / name, self.directory / 'source' / name)
            config['input_and_source_sha256']['source/' + name] = file_hash(root / name)
        config.update(
            model='radvl_aligned_covariance_CoVAE_weighted_graphs_residual_prediction',
            prediction_head=args.prediction_head,
            prediction_head_policy=dict(
                mode=args.prediction_head,
                residual_form='logit = s_AV + tanh(a) * s_fusion, with s_AV the Eq. 26 inner-product decoder '
                              'and s_fusion the Eq. 43 head; a is a single parameter initialised to zero',
                baseline='--prediction_head mlp reproduces the A3 model exactly (the residual parameter is not created)',
                warm_up='at a=0 the fusion head receives no gradient; it is trained through dL/da first, so the fusion '
                        'branch is switched on gradually. The fusion modules still receive gradient from L_align',
                supervision='with a=0 the prediction loss of Eq. 44 supervises the CoVAE latents directly, which is the '
                            'intended protection of the association signal',
                diagnostic_branches='residual mode adds fusion_only and association_plus_fusion to diagnostic_scores, so '
                                    'test_predictions.csv and branch_metrics.csv carry the four-way score comparison'),
            residual_scale_log='residual_scale.csv records residual_scale (=a), gamma = tanh(a) and the prediction-logit '
                               'scale for every epoch of every fold',
            evidence_notes=[
                'Fold-0 branch AUC from the released logs: CoVAE association inner product 0.8757, full fusion + MLP '
                '0.8806, fused embeddings by inner product 0.7532. The association view carries most of the signal and '
                'the fused vector is a poor inner-product embedding, which is what the residual form addresses.',
                'The residual form is not expected to deliver the +0.021 needed for 0.90 on its own; the fusion branch '
                'currently adds about +0.005 on fold 0, and blending two partially independent scores typically adds a '
                'few thousandths. Judge it with paired multi-seed differences against the same-fold A3 result.',
                'Run-to-run noise floor is at least 0.0018 mean test AUC (identical init, features, operator and first '
                'epoch, diverging from epoch 2 through CUDA non-determinism), so single-run comparisons are unusable.'],
            limitations=list(config.get('limitations', []))
                        + ['With a=0 the CoVAE score alone drives the prediction loss at initialisation; if gamma stays '
                           'near zero in the validation-selected checkpoint, the similarity view contributes nothing and '
                           'that is itself the finding to report.',
                           'This variant has not been trained by the editing task that created it.'])
        config['notes'].append('Residual prediction head recorded; --prediction_head mlp is the A3 control.')
        self.write_json('run_config.json', config)

    def epoch(self, fold, epoch, total, losses, model, optimizer, compute_seconds):
        super().epoch(fold, epoch, total, losses, model, optimizer, compute_seconds)
        if getattr(model, 'prediction_head', None) == 'residual_mlp':
            scale = float(model.residual_scale.detach().reshape(-1)[0])
            logits = model.prediction_logits.detach()
            append_csv(self.directory / 'residual_scale.csv', [dict(
                fold=fold, epoch=epoch, residual_scale=scale, gamma=float(torch.tanh(model.residual_scale).detach()),
                prediction_logit_std=logits.std(unbiased=False).item(),
                prediction_logit_abs_mean=logits.abs().mean().item())])

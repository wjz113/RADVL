"""RADVL recorder variants."""
import json
from pathlib import Path
import shutil

from radvl.recorder_diagnostics import RADVLResidualAuditRecorder
from radvl.data_protocol import file_hash


CONTROLS_SOURCE_FILES = ('objectives_variants.py',
                         'model_variants.py',
                         'settings_variants.py',
                         'recorder_variants.py',
                         'training_variants.py',
                         'test_model_options.py')


class RADVLResidualControlsRecorder(RADVLResidualAuditRecorder):
    def __init__(self, directory, args, k, dataset, protocol, requested_args):
        super().__init__(directory, args, k, dataset, protocol, requested_args)
        root = Path(__file__).resolve().parent
        config = json.loads((self.directory / 'run_config.json').read_text(encoding='utf-8'))
        for name in CONTROLS_SOURCE_FILES:
            shutil.copy2(root / name, self.directory / 'source' / name)
            config['input_and_source_sha256']['source/' + name] = file_hash(root / name)
        equivalent = dict(lambda_rec=args.lambda_av, beta_kl=args.lambda_av,
                          lambda_cov_star=args.lambda_av * args.lambda_cov)
        config.update(
            model='radvl_aligned_covariance_CoVAE_weighted_graphs_residual_prediction_balanced_'
                  'audited_controls',
            loss_coefficient_mode=args.loss_coefficients,
            coefficient_policy=dict(
                mode=args.loss_coefficients,
                grouped='lambda_av * (L_rec + L_KL + lambda_cov * Psi) computed as one grouped product, so the '
                        'total loss and the whole training trajectory are bit-identical to the entry that produced '
                        'the earlier results',
                decoupled='lambda_rec * L_rec + beta_kl * L_KL + lambda_cov_star * Psi, letting the three CoVAE '
                          'objectives be searched independently; algebraically identical to grouped with the '
                          'equivalent coefficients but not bit-identical, because the summation order changes',
                effective_coefficients=dict(reconstruction=(args.lambda_rec if args.loss_coefficients == 'decoupled'
                                                            else args.lambda_av),
                                            kl=(args.beta_kl if args.loss_coefficients == 'decoupled'
                                                else args.lambda_av),
                                            covariance=(args.lambda_cov_star
                                                        if args.loss_coefficients == 'decoupled'
                                                        else args.lambda_av * args.lambda_cov)),
                equivalent_of_the_grouped_baseline=equivalent),
            association_gradient=args.association_gradient,
            association_gradient_policy=(
                'logit = stopgrad(s_AV) + gamma * s_fusion: the association score keeps its forward value but the '
                'main prediction loss no longer reaches the CoVAE latents through s_AV, which isolates direct task '
                'supervision as the mechanism that prevents representation collapse. The CoVAE can still be reached '
                'indirectly through s_fusion and the alignment term.'
                if args.association_gradient == 'off' else
                'logit = s_AV + gamma * s_fusion with the association score carrying gradient (default).'),
            similarity_diagnostic=args.similarity_diagnostic,
            similarity_diagnostic_policy=(
                'similarity_inner_product = sigmoid(Z_sim_c @ Z_sim_d^T) is added to diagnostic_scores and therefore '
                'to branch_metrics.csv; Z_sim is captured by forward hooks on the two similarity projections. This is '
                'the unbiased read-out of the similarity view: fusion_only is contaminated because its input is the '
                'fused embedding first + second, whose second term is built from the association view.'
                if args.similarity_diagnostic == 'on' else 'disabled'),
            evidence_notes=[
                'Measured with the grouped mode: the covariance gradient ratio scales with lambda_cov over six '
                'points (6e-4 at lambda_cov=10 to 9.5e-2 at 3000) with a self-limiting numerator (ratio/lambda falls '
                '38-45% at fixed epochs while the prediction-gradient denominator stays flat at 0.46-0.54).',
                'Measured with the grouped mode: at matched epochs the cross-node geometry responds monotonically to '
                'lambda_cov (rho -5.7%, effective rank +12.5%, sigma -5.8% from lambda_cov=10 to 3000, endpoint '
                'direction consistent in 5/5 folds), while strict per-fold monotonicity holds only early (5/5 folds '
                'at epoch 20, 3/5 at 50, 1/5 at 80).',
                'Measured with the grouped mode: the posterior side is not untouched -- at matched epochs the mean '
                'absolute posterior mean falls 2.9-7.2% monotonically with lambda_cov, and the mean posterior '
                'variance and per-dimension KL move 0.3-4.1% and 0.8-3.1%.',
                'Measured head ablation: the ordinary MLP head collapses the association representation (effective '
                'rank 1.0-1.2, posterior variance 0.90-0.92, per-dimension KL 0.01) and costs about 0.03 test AUC; '
                'the residual head keeps the latents alive.'],
            limitations=list(config.get('limitations', []))
                        + ['The decoupled mode changes the summation order, so it is not bit-identical to the grouped '
                           'mode even with equivalent coefficients; compare it with a tolerance and never with '
                           'torch.equal.',
                           'This entry has not been trained by the editing task that created it.'])
        config['notes'].append('Coefficient mode, association-gradient and similarity-diagnostic policies recorded.')
        self.write_json('run_config.json', config)

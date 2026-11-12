"""RADVL settings variants."""
import argparse
import math


def parameter_parser():
    parser = argparse.ArgumentParser(
        description='RADVL-aligned CoVAE with the adaptive alignment temperature and a residual prediction head.')
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--protocol_dir', required=True)
    parser.add_argument('--scenario', choices=['edge', 'cold_circ', 'cold_drug', 'cold_both'], default='edge')
    parser.add_argument('--validation', type=int, default=5, help='Number of cross-validation folds.')
    parser.add_argument('--validation_fraction', type=float, default=0.1)
    parser.add_argument('--split_seed', type=int, default=114514)
    parser.add_argument('--seed', type=int, default=1234, help='Training random seed; data partitions are configured separately.')
    parser.add_argument('--epoch', type=int, default=800, help='Maximum number of training epochs.')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--min_delta', type=float, default=0.0)
    parser.add_argument('--mi_num', type=int, default=218, help='Number of drug nodes.')
    parser.add_argument('--dis_num', type=int, default=271, help='Number of circRNA-side nodes.')
    parser.add_argument('--train_fraction', type=float, default=1.0)
    parser.add_argument('--train_negative_ratio', type=float, default=1.0)
    parser.add_argument('--test_negative_ratio', type=float, default=1.0)
    parser.add_argument('--case_pairs', default=None)
    parser.add_argument('--case_drug_indices', type=int, nargs='*', default=[])
    parser.add_argument('--static_provenance', default=None, help='Optional existing provenance JSON; not required.')
    parser.add_argument('--prepare_only', action='store_true')
    parser.add_argument('--run_label', default='radvl_aligned_residual_prediction_validation_auc_earlystop')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--vae_hidden_dim', type=int, default=512)
    parser.add_argument('--hypergraph_width', type=int, default=256)
    parser.add_argument('--gin_layers', type=int, default=5)
    parser.add_argument('--hgcn_layers', type=int, default=4, help='Actual number of HGCN propagation layers.')
    parser.add_argument('--neighbors', type=int, default=8)
    parser.add_argument('--clusters', type=int, default=9)
    parser.add_argument('--prediction_head', choices=['residual_mlp', 'mlp'], default='residual_mlp',
                        help='residual_mlp (default): logit = s_AV + tanh(a)*s_fusion, so the CoVAE association score '
                             'is kept and the fusion branch earns its weight. mlp: the plain Eq. (43) head, which '
                             'recovers the A3 model exactly.')
    parser.add_argument('--residual_init', type=float, default=0.0,
                        help='Initial value of the residual scale parameter a, with gamma = tanh(a). 0.0 (default) '
                             'starts from the CoVAE association score alone; because dL/d s_fusion = gamma = 0 the '
                             'fusion head receives no gradient at step 0, and the measured runs then kept gamma at '
                             '0.003-0.017 while the fusion branch collapsed to AUC 0.5 in three of five folds. A '
                             'non-zero value such as 1.0 (gamma = 0.76) trains the fusion branch from the first step, '
                             'which is the fair test of whether it carries usable signal.')
    parser.add_argument('--loss_coefficients', choices=['grouped', 'decoupled'], default='grouped',
                        help='grouped (default): lambda_av * (L_rec + L_KL + lambda_cov * Psi) is computed as one '
                             'grouped product, so the total loss and the whole trajectory are bit-identical to the '
                             'entry that produced the current results. decoupled: lambda_rec * L_rec + beta_kl * L_KL '
                             '+ lambda_cov_star * Psi, so the three CoVAE objectives can be searched independently; '
                             'this is algebraically identical to grouped when the equivalent coefficients are used, '
                             'but not bit-identical, because the summation order changes.')
    parser.add_argument('--lambda_rec', type=float, default=None,
                        help='Weight of the reconstruction BCE when --loss_coefficients decoupled. Required in that '
                             'mode; the equivalent of the current grouped baseline (lambda_av) is printed when it '
                             'is missing.')
    parser.add_argument('--beta_kl', type=float, default=None,
                        help='Weight of the KL term when --loss_coefficients decoupled. Required in that mode; the '
                             'equivalent of the current grouped baseline (lambda_av) is printed when it is missing.')
    parser.add_argument('--lambda_cov_star', type=float, default=None,
                        help='Weight of the covariance penalty when --loss_coefficients decoupled. Required in that '
                             'mode; the equivalent of the current grouped baseline (lambda_av * lambda_cov) is '
                             'printed when it is missing.')
    parser.add_argument('--association_gradient', choices=['on', 'off'], default='on',
                        help='on (default): logit = s_AV + gamma * s_fusion with the association score carrying '
                             'gradient, so the main prediction loss supervises the CoVAE latents directly. off: '
                             'logit = stopgrad(s_AV) + gamma * s_fusion, which keeps the forward value but removes '
                             'that direct path, isolating it as the mechanism that prevents representation collapse.')
    parser.add_argument('--similarity_diagnostic', choices=['on', 'off'], default='on',
                        help='on (default): adds the detached similarity_inner_product branch (inner product on the '
                             'similarity-view embeddings captured by hooks) to diagnostic_scores and therefore to '
                             'branch_metrics.csv. This is the unbiased read-out of the similarity view, unlike '
                             'fusion_only whose input also contains the association view. off: skips it. NOTE: '
                             'measured, the branch is NOT observer-only -- it changes the memory-allocation pattern '
                             'inside the training forward, which flips the epoch-2 alignment bifurcation and costs '
                             'about 0.016 test AUC. Use off for any training whose numbers must be comparable with '
                             'the audit entry.')
    parser.add_argument('--lambda_align_warmup', type=int, default=0,
                        help='0 (default): lambda_align applies at full value from the first epoch. N > 0: '
                             'lambda_align(t) = lambda_align * min(1, t / N) with t the 1-based epoch index inside '
                             'each fold. Motivation: the alignment loss bifurcates at epoch 2 (about 105 against '
                             'about 700 depending on sub-1e-8 numerical differences) and the branch decides about '
                             '0.016 of test AUC and the latent effective rank (4.1 against 1.6).')
    parser.add_argument('--gin_weighting', choices=['uniform', 'similarity'], default='uniform',
                        help='uniform (default, current behaviour): unweighted neighbour sum of Eq. (15). '
                             'similarity: scale each neighbour message by the fused similarity stored in edge_attr, '
                             'which Eq. (15) does not use at all.')
    parser.add_argument('--gin_weight_normalization', choices=['mass', 'mean'], default='mass',
                        help='mass (default): rescale the similarity weights so each receiver keeps the same total '
                             'message mass as the unweighted sum, isolating the redistribution of mass. '
                             'mean: weighted average of neighbours.')
    parser.add_argument('--hyperedge_weight', choices=['unit', 'similarity'], default='unit',
                        help='unit (default, current behaviour): W = I in Eq. (17)-(19). similarity: w_e = mean '
                             'intra-hyperedge similarity, i.e. the weight matrix W that the manuscript denotes but '
                             'never defines.')
    parser.add_argument('--hyperedge_weight_rescale', choices=['on', 'off'], default='on',
                        help='on (default): rescale w to unit mean so the operator scale stays comparable to the '
                             'unit-weight case; off: use the raw mean intra-hyperedge similarity.')
    parser.add_argument('--fusion_tau_mode', choices=['fixed', 'adaptive'], default='adaptive',
                        help='adaptive (default for this entry): tau = fusion_tau_scale * mean(D) per forward pass, '
                             'because D is O(1e4) at initialisation and O(1e3) after convergence, which makes a fixed '
                             'tau=100 drive omega below 1e-8 for more than 90% of the nodes. fixed: constant --fusion_tau.')
    parser.add_argument('--fusion_tau_scale', type=float, default=1.0,
                        help='Multiplier of mean(D) when --fusion_tau_mode adaptive; 1.0 means tau equals the mean '
                             'discrepancy.')
    parser.add_argument('--fusion_tau', type=float, default=100.0, help='Constant alignment temperature when --fusion_tau_mode fixed.')
    parser.add_argument('--lambda_cov', '--co_lambda', dest='lambda_cov', type=float, default=0.5)
    parser.add_argument('--lambda_av', type=float, default=1.0, help='lambda_1 in Eq. (45).')
    parser.add_argument('--lambda_align', '--fusion_alignment_weight', dest='lambda_align', type=float, default=2.0)
    parser.add_argument('--eta_pos', type=float, default=1.0)
    parser.add_argument('--eta_neg', type=float, default=1.0)
    parser.add_argument('--pair_hidden_dim', type=int, default=64)
    parser.add_argument('--pair_chunk_rows', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    for name in ['validation', 'epoch', 'patience', 'mi_num', 'dis_num', 'hid_dim', 'latent_dim',
                 'vae_hidden_dim', 'hypergraph_width', 'gin_layers', 'hgcn_layers', 'neighbors',
                 'clusters', 'pair_hidden_dim', 'pair_chunk_rows']:
        if getattr(args, name) < 1:
            parser.error(name + ' must be positive')
    if args.validation < 2 or args.latent_dim < 2:
        parser.error('validation and latent_dim must be at least 2')
    if args.hypergraph_width < 4:
        parser.error('hypergraph_width must be at least 4 for the retained HGCN recalibration block')
    if not 0 < args.validation_fraction < 1 or not 0 < args.train_fraction <= 1:
        parser.error('Require 0 < validation_fraction < 1 and 0 < train_fraction <= 1')
    if not 0 <= args.dropout < 1:
        parser.error('dropout must be in [0,1)')
    for name in ['min_delta', 'lambda_cov', 'lambda_av', 'lambda_align', 'weight_decay']:
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(name + ' must be finite and nonnegative')
    for name in ['fusion_tau', 'fusion_tau_scale', 'learning_rate', 'eta_pos', 'eta_neg',
                 'train_negative_ratio', 'test_negative_ratio']:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(name + ' must be finite and positive')
    if not 0 <= args.seed < 2**32 or not 0 <= args.split_seed < 2**32 - 10000 - args.validation**2:
        parser.error('Seeds must fit the RNG range, including protocol per-fold offsets')
    if args.lambda_align_warmup < 0:
        parser.error('lambda_align_warmup must be nonnegative')
    if args.loss_coefficients == 'decoupled':
        equivalent = dict(lambda_rec=args.lambda_av, beta_kl=args.lambda_av,
                          lambda_cov_star=args.lambda_av * args.lambda_cov)
        for name in ('lambda_rec', 'beta_kl', 'lambda_cov_star'):
            value = getattr(args, name)
            if value is None:
                parser.error('--loss_coefficients decoupled requires --%s (the equivalent of the current '
                             'grouped baseline is %s = %.6g)' % (name, name, equivalent[name]))
            if not math.isfinite(value) or value < 0:
                parser.error(name + ' must be finite and nonnegative')
    # Compatibility for the existing recorder only; the objective uses lambda_align.
    args.fusion_alignment_weight = args.lambda_align
    return args

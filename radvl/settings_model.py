"""RADVL settings model."""
import argparse
import math


def parameter_parser():
    parser = argparse.ArgumentParser(description='RADVL-aligned CoVAE and undirected GIN, with validation-AUC early stopping.')
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
    parser.add_argument('--run_label', default='radvl_aligned_validation_auc_earlystop')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--vae_hidden_dim', type=int, default=512)
    parser.add_argument('--hypergraph_width', type=int, default=256)
    parser.add_argument('--gin_layers', type=int, default=5)
    parser.add_argument('--hgcn_layers', type=int, default=4, help='Actual number of HGCN propagation layers.')
    parser.add_argument('--neighbors', type=int, default=8)
    parser.add_argument('--clusters', type=int, default=9)
    parser.add_argument('--fusion_tau', type=float, default=100.0, help='Alignment temperature in Eq. (41); not a tuned optimum.')
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
    for name in ['fusion_tau', 'learning_rate', 'eta_pos', 'eta_neg', 'train_negative_ratio', 'test_negative_ratio']:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(name + ' must be finite and positive')
    if not 0 <= args.seed < 2**32 or not 0 <= args.split_seed < 2**32 - 10000 - args.validation**2:
        parser.error('Seeds must fit the RNG range, including protocol per-fold offsets')
    # Compatibility for the existing recorder only; the objective uses lambda_align.
    args.fusion_alignment_weight = args.lambda_align
    return args

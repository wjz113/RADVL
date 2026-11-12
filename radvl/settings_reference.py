"""RADVL settings reference."""
import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--validation', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=800)
    # Legacy names: mi_num is drug count, dis_num is circRNA count.
    parser.add_argument('--mi_num', type=int, default=218)
    parser.add_argument('--dis_num', type=int, default=271)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--co_lambda', '--lambda_cov', dest='co_lambda',
                        type=float, default=0.5)
    parser.add_argument('--lambda_av', type=float, default=1.0)
    parser.add_argument('--lambda_align', type=float, default=2.0)
    parser.add_argument('--align_tau', type=float, default=0.05)
    parser.add_argument('--eta_pos', type=float, default=1.0)
    parser.add_argument('--eta_neg', type=float, default=1.0)
    args = parser.parse_args()
    if args.align_tau <= 0:
        parser.error('--align_tau must be positive')
    if min(args.co_lambda, args.lambda_av, args.lambda_align,
           args.eta_pos, args.eta_neg) < 0:
        parser.error('loss weights must be nonnegative')
    return args

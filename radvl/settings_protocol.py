import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description="Run RADVL.")

    parser.add_argument('--data_path', type=str, default='./data',
                        help='Directory containing the RADVL input matrices.')
    parser.add_argument('--validation', type=int, default=5,
                        help='Number of cross-validation folds.')
    # parser.add_argument('--epoch', type=int, default=350,
    parser.add_argument('--epoch', type=int, default=800,  #800
                        help='Maximum number of training epochs.')

    parser.add_argument('--mi_num', type=int, default=218,
                        help='Number of drug nodes.')
    parser.add_argument('--dis_num', type=int, default=271,
                        help='Number of circRNA-side nodes.')

    parser.add_argument('--alpha', type=float, default=0.11,
                        help='Coefficient alpha.')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability.')
    # parser.add_argument('--nlayer', type=int, default=2,
    #                     help='Number of layers.')
    parser.add_argument('--n_hidden', type=int, default=20,
                        help='Hidden feature dimension.')
    parser.add_argument('--nmodal', type=int, default=2,
                        help='Number of input views.')


    parser.add_argument('--hid_dim', type=int, default=64,
                        help='Hidden representation dimension.')

    parser.add_argument('--item', type=int, default=218,
                        help='Number of drug nodes.')
    parser.add_argument('--user', type=int, default=271,
                        help='Number of circRNA-side nodes.')

    # Covariance-regularized variational encoder
    # D-CoVAE 超参数：总体强度 co_lambda，softmax 温度 co_tau
    parser.add_argument('--co_lambda', type=float, default=5e-1)

    parser.add_argument('--co_tau', type=float, default=0.05)


    parser.add_argument('--seed', type=int, default=1234,
                        help='Training random seed; data partitions are configured separately.')
    parser.add_argument('--run_label', type=str, default='original_architecture_shared_protocol',
                        help='Optional label for a local RADVL run.')
    parser.add_argument('--split_seed', type=int, default=114514)
    parser.add_argument('--case_pairs', type=str, default=None,
                        help='CSV with zero-based circ_index,drug_index; all candidate unknowns excluded from training.')
    parser.add_argument('--case_drug_indices', type=int, nargs='*', default=[],
                        help='Reserve all unknown pairs for these zero-based drug columns for case ranking.')
    parser.add_argument('--protocol_dir', type=str, required=True,
                        help='Frozen split directory shared with all baseline models.')
    parser.add_argument('--scenario', choices=['edge','cold_circ','cold_drug','cold_both'], default='edge')
    parser.add_argument('--train_fraction', type=float, default=1.0)
    parser.add_argument('--train_negative_ratio', type=float, default=1.0)
    parser.add_argument('--test_negative_ratio', type=float, default=1.0)
    parser.add_argument('--validation_fraction', type=float, default=0.0)
    parser.add_argument('--static_provenance', type=str, default=None,
                        help='Optional JSON recording association-independent static feature provenance.')
    parser.add_argument('--prepare_only', action='store_true', help='Create/audit the protocol without model training.')
    args = parser.parse_args()
    if args.validation < 2:
        parser.error('--validation must be at least 2 for out-of-fold evaluation')
    return args



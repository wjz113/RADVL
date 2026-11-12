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

    parser.add_argument('--alpha', type=int, default=0.11,
                        help='Coefficient alpha.')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability.')
    # parser.add_argument('--nlayer', type=int, default=2,
    #                     help='Number of layers.')
    parser.add_argument('--n_hidden', type=int, default=20,
                        help='Hidden feature dimension.')
    parser.add_argument('--nmodal', type=int, default=2,
                        help='Number of input views.')


    parser.add_argument('--latdim', type=int, default=64,
                        help='Number of views.')

    parser.add_argument('--item', type=int, default=218,
                        help='Number of drug nodes.')
    parser.add_argument('--user', type=int, default=271,
                        help='Number of circRNA-side nodes.')

    return parser.parse_args()
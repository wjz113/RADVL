"""RADVL training cross validation."""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# The gradient audit (diagnostics only, several extra backward passes per audited epoch) is on
# by default in the shared code (every 5 epochs) and roughly halves the training speed.  All
# reported HAP results were produced with it off; set GRADIENT_AUDIT_EVERY=5 to re-enable it.
os.environ.setdefault('GRADIENT_AUDIT_EVERY', '0')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '2')

RESULTS_ROOT = Path('Results_dadvl_hap_cv5_712')
LOGS = Path('_cv5_712_logs')
PROTOCOL, VALIDATION_FRACTION = 'edge_val125', 0.125
METRICS = ['AUC', 'AUPR', 'F1', 'Accuracy', 'Recall', 'Specificity', 'Precision']


def warmup_stopping(base, min_epochs):
    """Validation-AUC early stopping whose selection window starts at ``min_epochs``.

    With the HAP prior the epoch-1 validation AUC is already high (the prior alone scores about
    0.89-0.91); the model then dips and needs about 60 epochs to recover, so plain patience-40
    stopping restores epoch 1.  Epochs before ``min_epochs`` are logged but can neither be
    selected nor consume patience.
    """
    class WarmupAucEarlyStopping(base):
        def update(self, epoch, auc):
            if epoch < min_epochs:
                return dict(is_best=False, patience_reset=False, should_stop=False)
            return super().update(epoch, auc)

    return WarmupAucEarlyStopping


def run_single(split_seed, own, extra):
    """Train one split in this process (called through ``--_single``)."""
    from radvl import training_rah as hyper
    hyper._WEIGHTED.RESULTS_ROOT = str(RESULTS_ROOT)
    v2 = hyper.fixed11.v2
    v2.AucEarlyStopping = warmup_stopping(v2.AucEarlyStopping, own.min_epochs)
    argv = ['--protocol_dir', f'protocols/{PROTOCOL}_seed{split_seed}', '--split_seed', str(split_seed),
            '--validation_fraction', str(VALIDATION_FRACTION), '--seed', str(own.seed),
            '--ema_decay', str(own.ema_decay)] + list(extra)
    hap_args, remaining = hyper.split_arguments(argv)
    hyper.SETTINGS.update(vars(hap_args))
    fixed_args, remaining = hyper.fixed11.split_arguments(remaining)
    sys.argv = [sys.argv[0]] + remaining
    parsed = hyper.controls.parameter_parser()
    for key, value in vars(hap_args).items():
        setattr(parsed, key, value)
    parsed.min_epochs = own.min_epochs
    parsed.save_checkpoints = globals().get('SAVE_CHECKPOINTS', 'on')
    hyper.fixed11.main(parsed, fixed_args)


def run_and_tee(command, log):
    """Run ``command``, copying its combined output live to the terminal and to ``log``.

    The output is forwarded in raw chunks rather than lines, so the per-fold tqdm progress bar
    (which redraws with carriage returns) stays live in the terminal.
    """
    terminal = sys.stdout.buffer
    with open(log, 'wb') as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            chunk = process.stdout.read1(4096)
            if not chunk:
                break
            terminal.write(chunk)
            terminal.flush()
            handle.write(chunk)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def result_directory(log):
    for line in log.read_text(errors='ignore').splitlines():
        if line.startswith('Results:'):
            path = re.search(r'(Results_dadvl_hap_cv5_712\S+)', line).group(1)
            return Path(path.replace(chr(92), '/'))
    raise RuntimeError(f'No results directory in {log}')


def summarise(rows, seed):
    frame = pd.DataFrame(rows)
    epochs = [int(e) for text in frame['best_epochs'] for e in text.split()]
    mean = dict(split_seed='mean', seed=seed, **frame[METRICS + ['validation_AUC']].mean().to_dict(),
                best_epochs=f'median {int(pd.Series(epochs).median())}')
    frame = pd.concat([frame, pd.DataFrame([mean])], ignore_index=True)
    out = RESULTS_ROOT / f'summary_seed{seed}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print('OVERALL TEST MEAN: ' + ' | '.join(f'{m}={mean[m]:.4f}' for m in METRICS), flush=True)
    print(f"Mean validation AUC over all folds (compare hyperparameters on this): {mean['validation_AUC']:.4f}; "
          f"best epoch {mean['best_epochs']}", flush=True)
    print('summary:', out.resolve(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split_seeds', type=int, nargs='+', required=True)
    parser.add_argument('--seed', type=int, required=True, help='Training random seed; data partitions are configured separately.')
    parser.add_argument('--ema_decay', type=float, required=True)
    parser.add_argument('--min_epochs', type=int, required=True, help='First epoch eligible for selection.')
    parser.add_argument('--_single', type=int, default=None, help=argparse.SUPPRESS)
    own, extra = parser.parse_known_args()
    banned = {'--protocol_dir', '--split_seed', '--validation_fraction', '--fixed_epochs', '--stage'} & set(extra)
    if banned:
        parser.error(f'{sorted(banned)} are not used: the protocol is fixed to 7:1:2 with validation early stopping')
    if own._single is not None:
        run_single(own._single, own, extra)
        return
    LOGS.mkdir(exist_ok=True)
    rows = []
    for split_seed in own.split_seeds:
        log = LOGS / f'split{split_seed}_seed{own.seed}.log'
        print(f'split {split_seed} ... log: {log}', flush=True)
        run_and_tee([sys.executable, '-u', __file__, '--_single', str(split_seed),
                     '--split_seeds', str(split_seed), '--seed', str(own.seed),
                     '--ema_decay', str(own.ema_decay), '--min_epochs', str(own.min_epochs), *extra], log)
        directory = result_directory(log)
        metrics = pd.read_csv(directory / 'fold_metrics.csv')
        selection = pd.read_csv(directory / 'early_stopping_summary.csv')
        row = dict(split_seed=split_seed, seed=own.seed, **{m: metrics[m].mean() for m in METRICS},
                   validation_AUC=selection['best_validation_AUC'].mean(),
                   best_epochs=' '.join(map(str, selection['best_epoch'])))
        rows.append(row)
        print(f'split {split_seed}: ' + ' | '.join(f'{m}={row[m]:.4f}' for m in METRICS)
              + f" | validation_AUC={row['validation_AUC']:.4f} | best epochs {row['best_epochs']}", flush=True)
    summarise(rows, own.seed)


if __name__ == '__main__':
    main()

"""RADVL train entry."""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ['GRADIENT_AUDIT_EVERY'] = '0'
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '2')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parameters', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split_seed', type=int, required=True)
    p.add_argument('--save_checkpoints', choices=['on', 'off'], default='off')
    a = p.parse_args()
    parameters = json.loads(Path(a.parameters).read_text(encoding='utf-8'))
    from radvl import training_relation_gating as relation_gating_controls
    from radvl import model_options as review
    relation_controls = relation_gating_controls.relation_controls
    masking_controls = relation_controls.masking_controls
    context_controls = masking_controls.context_controls
    token_controls = context_controls.token_controls
    readout_controls = token_controls.readout_controls
    variance_controls = readout_controls.variance_controls
    discrepancy_controls = variance_controls.discrepancy_controls
    for module in (readout_controls, token_controls, context_controls, masking_controls, relation_controls, relation_gating_controls, discrepancy_controls, variance_controls, review):
        module.configure({k: parameters.pop(k) for k in list(module.DEFAULTS) if k in parameters})
    from radvl.model import RADVL
    protocol_name = parameters.pop('protocol_name', 'edge_val125')
    from radvl import training_cross_validation as original
    original.PROTOCOL = protocol_name
    original.RESULTS_ROOT = Path(a.output).resolve()
    original.SAVE_CHECKPOINTS = a.save_checkpoints
    own = SimpleNamespace(seed=parameters.pop('seed'),
                          min_epochs=parameters.pop('min_epochs'),
                          ema_decay=parameters.pop('ema_decay'))
    extra = []
    for key, value in parameters.items():
        if isinstance(value, list):
            extra.extend(['--' + key] + [str(item) for item in value])
        else:
            extra.extend(['--' + key, str(value)])
    original.run_single(a.split_seed, own, extra)


if __name__ == '__main__':
    main()

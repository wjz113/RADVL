"""RADVL training relations."""
import numpy as np
import torch

from radvl import training_masking as masking_controls

context_controls = masking_controls.context_controls
token_controls = context_controls.token_controls
readout_controls = token_controls.readout_controls
discrepancy_controls = token_controls.discrepancy_controls
controls = masking_controls.controls
_WEIGHTED = masking_controls._WEIGHTED
_MASKING_MODEL = _WEIGHTED.RADVL
DEFAULTS = dict(edge_mask=0.0)
SETTINGS = dict(DEFAULTS)


def configure(values):
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown DFX8 settings: {sorted(unknown)}')
    merged = dict(DEFAULTS, **values)
    merged['edge_mask'] = float(merged['edge_mask'])
    if not 0 <= merged['edge_mask'] < 1:
        raise ValueError('edge_mask must be in [0, 1)')
    SETTINGS.clear()
    SETTINGS.update(merged)


# ---------------------------------------------------------------- shuffled-discrepancy gate control
discrepancy_controls.CHOICES['gate_mode'] = tuple(discrepancy_controls.CHOICES['gate_mode']) + ('absdelta_shuffled',)
_ORIGINAL_GATE_SIGNAL = discrepancy_controls.gate_signal
_PERMUTATIONS = {}


def _derangement(n):
    if n not in _PERMUTATIONS:
        order = np.random.default_rng([8191, n]).permutation(n)
        perm = np.empty(n, dtype=np.int64)
        perm[order] = np.roll(order, -1)
        _PERMUTATIONS[n] = torch.as_tensor(perm)
    return _PERMUTATIONS[n]


def gate_signal(delta, mode):
    if mode == 'absdelta_shuffled':
        return delta.abs()[_derangement(len(delta)).to(delta.device)]
    return _ORIGINAL_GATE_SIGNAL(delta, mode)


discrepancy_controls.gate_signal = gate_signal


# ---------------------------------------------------------------- masked supervision
class RADVL(_MASKING_MODEL):
    def __init__(self, n_circ, n_drug, args):
        super().__init__(n_circ, n_drug, args)
        self.dfx8_mask = SETTINGS['edge_mask']
        self.dfx8_generator = None
        self.dfx8_positive = self.dfx8_negative = None
        self.dfx8_supervised = None

    def forward(self, A, *rest, **kwargs):
        self.dfx8_supervised = None
        if self.training and self.dfx8_mask > 0:
            if self.dfx8_generator is None:
                raise RuntimeError('edge mask generator was not bound')
            g, pos, neg = self.dfx8_generator, self.dfx8_positive, self.dfx8_negative
            chosen = torch.rand(len(pos), generator=g, device=pos.device) < self.dfx8_mask
            if not chosen.any():
                chosen[torch.randint(len(pos), (1,), generator=g, device=pos.device)] = True
            supervised = pos[chosen]
            negatives = neg[torch.randperm(len(neg), generator=g, device=neg.device)[:len(supervised)]]
            A = A.clone()
            A[supervised[:, 0], supervised[:, 1]] = 0
            self.dfx8_supervised = (supervised, negatives)
        return super().forward(A, *rest, **kwargs)


class RelationObjective:
    def __init__(self, base_objective):
        self.base_objective = base_objective

    def __call__(self, model, reconstruction_logits, positive_pairs, negative_pairs, covariance, alignment, args):
        selected = getattr(model, 'dfx8_supervised', None)
        if model.training and selected is not None:
            positive_pairs, negative_pairs = selected
        return self.base_objective(model, reconstruction_logits, positive_pairs, negative_pairs, covariance, alignment, args)


_PREVIOUS_FIT = _WEIGHTED.fit


def _fit(model, optimizer, training, validation, graphs, args, recorder):
    if model.dfx8_mask > 0:
        device = graphs['A'].device
        model.dfx8_positive = torch.as_tensor(np.asarray(training.positive_pairs), dtype=torch.long, device=device)
        model.dfx8_negative = torch.as_tensor(np.asarray(training.negative_pairs), dtype=torch.long, device=device)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(args.seed) * 1000003 + int(training.fold) * 31 + 11)
        model.dfx8_generator = generator
    return _PREVIOUS_FIT(model, optimizer, training, validation, graphs, args, recorder)


_WEIGHTED.fit = _fit
_WEIGHTED.RADVL = RADVL
controls._CONTROLS_OBJECTIVE = RelationObjective(controls._CONTROLS_OBJECTIVE)
_DFX7_PARSER = controls.parameter_parser


def _parser_with_relations():
    parsed = _DFX7_PARSER()
    for key, value in SETTINGS.items():
        setattr(parsed, key, value)
    return parsed


controls.parameter_parser = _parser_with_relations

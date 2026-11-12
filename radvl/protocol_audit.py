"""RADVL protocol audit."""
import numpy as np

from radvl import data_protocol as sep

TOLERATED = ('train_test_overlap', 'heldout_pairs_in_train_mask')
RECORDED = ('test_positives_in_train_negative', 'test_negatives_in_train_negative')
_ORIGINAL = sep.SharedProtocol.audit


def is_label_blind(protocol):
    return protocol.manifest.get('derived_from', {}).get('strategy') == 'label_blind'


def _pairs(array):
    return set(map(tuple, np.asarray(array)))


def audit(self, fold):
    if not is_label_blind(self):
        return _ORIGINAL(self, fold)
    values, train = self.arrays(fold), self.training(fold)
    groups = {part: _pairs(np.vstack([values[part + '_positive'], values[part + '_negative']]))
              for part in ('train', 'validation', 'test')}
    result = dict(fold=fold, train_validation_overlap=len(groups['train'] & groups['validation']),
                  train_test_overlap=len(groups['train'] & groups['test']),
                  validation_test_overlap=len(groups['validation'] & groups['test']))
    validation = np.vstack([values['validation_positive'], values['validation_negative']])
    test = np.vstack([values['test_positive'], values['test_negative']])
    in_mask_validation = int(train.train_mask[validation[:, 0], validation[:, 1]].sum())
    in_mask_test = int(train.train_mask[test[:, 0], test[:, 1]].sum())
    result['heldout_pairs_in_train_mask'] = in_mask_validation + in_mask_test
    result['validation_pairs_in_train_mask'] = in_mask_validation
    heldout_positive = np.vstack([values['validation_positive'], values['test_positive']])
    result['heldout_positives_in_A_train'] = int(train.A_train[heldout_positive[:, 0], heldout_positive[:, 1]].sum())
    result['candidate_negative_overlap'] = len(_pairs(values['train_negative']) & _pairs(self.candidates))
    heldout = np.vstack([validation, test])
    result['heldout_pairs_in_allowed_negative_pool'] = int(
        train.allowed_negative_mask[heldout[:, 0], heldout[:, 1]].sum())
    negatives = _pairs(values['train_negative'])
    result['test_positives_in_train_negative'] = len(negatives & _pairs(values['test_positive']))
    result['test_negatives_in_train_negative'] = len(negatives & _pairs(values['test_negative']))
    for name, value in result.items():
        if name in ('fold',) + TOLERATED + RECORDED:
            continue
        if value != 0:
            raise AssertionError(f'Isolation violation under label-blind sampling: {result}')
    if result['train_test_overlap'] != result['test_positives_in_train_negative'] + result['test_negatives_in_train_negative']:
        raise AssertionError(f'Training/test overlap is not explained by training negatives: {result}')
    result.update({name + '_count': len(values[name]) for name in sep.PAIR_KEYS})
    result['training_fingerprint'] = train.fingerprint
    return result


def install():
    sep.SharedProtocol.audit = audit

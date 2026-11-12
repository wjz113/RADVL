"""RADVL data preprocessing."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def gip(profiles):
    profiles = profiles.astype(np.float64)
    norms = np.square(profiles).sum(axis=1)
    scale = norms.mean()
    if scale == 0:
        return np.eye(len(profiles), dtype=np.float64)
    distances = np.maximum(norms[:, None] + norms[None, :] - 2*profiles @ profiles.T, 0)
    return np.exp(-distances / scale)


def entropy_similarity(profiles):
    """Degree-weighted overlap entropy, fully determined by the training profiles.

    p_j = training_degree(j)/total_training_edges; w_j=-p_j log2(p_j).
    S_ij=2*sum_shared(w)/(sum_profile_i(w)+sum_profile_j(w)).
    Zero denominators give zero; diagonal is one for nonempty weighted profiles.
    These are globally normalized degree weights, not per-profile probabilities.
    """
    profiles = (profiles > 0).astype(np.float64)
    degrees = profiles.sum(axis=0)
    total = degrees.sum()
    weights = np.zeros_like(degrees)
    if total:
        probabilities = degrees / total
        valid = probabilities > 0
        weights[valid] = -probabilities[valid] * np.log2(probabilities[valid])
    entropy = profiles @ weights
    common = (profiles * weights) @ profiles.T
    denominator = entropy[:, None] + entropy[None, :]
    similarity = np.divide(2*common, denominator, out=np.zeros_like(common), where=denominator > 0)
    return np.clip(similarity, 0, 1)


def read_static(path, identifiers):
    table = pd.read_csv(path, index_col=0)
    table.index = table.index.astype(str)
    table.columns = table.columns.astype(str)
    identifiers = list(map(str, identifiers))
    if not table.index.is_unique or not table.columns.is_unique:
        raise ValueError(f'Duplicate static feature IDs in {path}')
    if set(table.index) != set(identifiers) or set(table.columns) != set(identifiers):
        raise ValueError(f'Static feature IDs do not match association ordering: {path}')
    values = table.loc[identifiers, identifiers].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1+1e-6).any():
        raise ValueError(f'Static similarities must be finite in [0,1]: {path}')
    return values


def build_fold(positive_pairs, negative_pairs, test_positive, test_negative,
               shape, circ_static, drug_static, candidates):
    train_A = np.zeros(shape, dtype=np.float32)
    train_A[positive_pairs[:, 0], positive_pairs[:, 1]] = 1
    reconstruction_mask = np.zeros(shape, dtype=bool)
    train_pairs = np.vstack([positive_pairs, negative_pairs])
    reconstruction_mask[train_pairs[:, 0], train_pairs[:, 1]] = True
    heldout = np.vstack([test_positive, test_negative])
    if train_A[test_positive[:, 0], test_positive[:, 1]].any():
        raise AssertionError('Test positive leaked into association inputs')
    if reconstruction_mask[heldout[:, 0], heldout[:, 1]].any():
        raise AssertionError('Held-out pair used as reconstruction supervision')
    if len(candidates):
        negative_set = set(map(tuple, negative_pairs))
        if negative_set & set(map(tuple, candidates)):
            raise AssertionError('Case candidate resampled as a training negative')
    circ_gip, drug_gip = gip(train_A), gip(train_A.T)
    circ_entropy, drug_entropy = entropy_similarity(train_A), entropy_similarity(train_A.T)
    # Explicit documented fusion; no precomputed full-A integration files.
    circ_similarity = np.where(circ_static > 0,
                               (circ_static + circ_gip + circ_entropy)/3, circ_gip)
    drug_similarity = np.where(drug_static > 0,
                               (drug_static + drug_gip + drug_entropy)/3, drug_gip)
    return dict(A=torch.from_numpy(train_A), reconstruction_mask=torch.from_numpy(reconstruction_mask),
                IM=torch.tensor(circ_similarity, dtype=torch.float32),
                ID=torch.tensor(drug_similarity, dtype=torch.float32),
                circ_gip=circ_gip, drug_gip=drug_gip,
                circ_entropy=circ_entropy, drug_entropy=drug_entropy)


def prepare_data(opt):
    root = Path(opt.data_path)
    association = pd.read_csv(root / 'association.csv', index_col=0)
    truth = association.to_numpy(dtype=np.float32)
    if not np.isfinite(truth).all() or not np.isin(truth, [0, 1]).all():
        raise ValueError('association.csv must be a finite binary matrix')
    if truth.shape != (opt.dis_num, opt.mi_num):
        raise ValueError('Association shape must match dis_num rows and mi_num columns')
    circ_static = read_static(root / 'gene_seq_sim.csv', association.index)
    drug_static = read_static(root / 'drug_str_sim.csv', association.columns)
    candidates = np.empty((0, 2), dtype=np.int64)
    if opt.case_pairs:
        table = pd.read_csv(opt.case_pairs)
        if not {'circ_index', 'drug_index'}.issubset(table.columns):
            raise ValueError('--case_pairs needs circ_index and drug_index columns')
        raw = table[['circ_index', 'drug_index']].to_numpy(dtype=np.float64)
        if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
            raise ValueError('Candidate pair indices must be finite integers')
        candidates = raw.astype(np.int64)
    for drug in opt.case_drug_indices:
        if not 0 <= drug < truth.shape[1]:
            raise ValueError('Case drug index out of range')
        candidates = np.vstack([candidates, np.column_stack([
            np.arange(truth.shape[0]), np.full(truth.shape[0], drug)])])
    if len(candidates):
        if ((candidates < 0).any() or (candidates[:, 0] >= truth.shape[0]).any()
                or (candidates[:, 1] >= truth.shape[1]).any()):
            raise ValueError('Candidate pair index out of range')
        candidates = np.unique(candidates, axis=0)
    candidate_mask = np.zeros_like(truth, dtype=bool)
    candidate_mask[candidates[:, 0], candidates[:, 1]] = True
    # Local RNG; feature construction never reads test labels.
    rng = np.random.RandomState(opt.split_seed)
    positives = np.argwhere(truth == 1)
    unknowns = np.argwhere((truth == 0) & ~candidate_mask)
    if len(unknowns) < len(positives):
        raise ValueError('Not enough nonreserved unknown pairs for balanced CV')
    if len(positives) < opt.validation:
        raise ValueError('Too few positives for requested folds')
    rng.shuffle(unknowns)
    rng.shuffle(positives)
    sampled_negatives = unknowns[:len(positives)]
    pos_folds = np.array_split(positives, opt.validation)
    neg_folds = np.array_split(sampled_negatives, opt.validation)
    dataset = dict(md=[], folds=[], md_true=torch.from_numpy(truth),
                   candidates=candidates, association=association,
                   circ_static=circ_static, drug_static=drug_static,
                   static_files=['gene_seq_sim.csv', 'drug_str_sim.csv'])
    for fold in range(opt.validation):
        train_pos = np.concatenate([p for i, p in enumerate(pos_folds) if i != fold])
        train_neg = np.concatenate([p for i, p in enumerate(neg_folds) if i != fold])
        dataset['md'].append(dict(
            train=[torch.from_numpy(train_pos), torch.from_numpy(train_neg)],
            test=[torch.from_numpy(pos_folds[fold]), torch.from_numpy(neg_folds[fold])]))
        # Fold construction is lazy; no full truth enters model data fields.
        dataset['folds'].append(None)
    return dataset


class Dataset:
    def __init__(self, opt, dataset):
        self.opt, self.dataset = opt, dataset

    def __getitem__(self, fold):
        dataset = self.dataset
        split = dataset['md'][fold]
        if dataset['folds'][fold] is None:
            dataset['folds'][fold] = build_fold(
                split['train'][0].numpy(), split['train'][1].numpy(),
                split['test'][0].numpy(), split['test'][1].numpy(),
                dataset['md_true'].shape, dataset['circ_static'], dataset['drug_static'],
                dataset['candidates'])
        data = dataset['folds'][fold]
        # Field 5 is evaluation truth only. Fields 4/6 are strictly fold training A.
        return (data['ID'], data['IM'], split['train'], split['test'],
                data['A'], dataset['md_true'], data['A'], data['reconstruction_mask'])

    def __len__(self):
        return self.opt.validation

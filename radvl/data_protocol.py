"""RADVL data protocol."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA = 1
PAIR_KEYS = ('train_positive', 'train_negative', 'validation_positive',
             'validation_negative', 'test_positive', 'test_negative')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    payload = str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()
    return hashlib.sha256(payload).hexdigest()


def readonly(array):
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TrainingFold:
    fold: int
    A_train: np.ndarray
    positive_pairs: np.ndarray
    negative_pairs: np.ndarray
    train_mask: np.ndarray
    allowed_negative_mask: np.ndarray
    fingerprint: str


@dataclass(frozen=True)
class EvaluationSet:
    pairs: np.ndarray
    labels: np.ndarray


class SharedProtocol:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['schema'] != SCHEMA:
            raise ValueError('Unsupported protocol schema')
        for name, digest in self.manifest['artifacts'].items():
            if file_hash(self.directory / name) != digest:
                raise ValueError(f'Frozen protocol artifact changed: {name}')
        self.shape = tuple(self.manifest['shape'])
        self.circ_ids = self.manifest['circ_ids']
        self.drug_ids = self.manifest['drug_ids']
        self.candidates = readonly(np.load(self.directory / 'candidates.npy', allow_pickle=False))

    def __len__(self):
        return self.manifest['fit_count']

    def arrays(self, fold):
        with np.load(self.directory / f'fold_{fold}.npz', allow_pickle=False) as data:
            return {name: data[name].copy() for name in PAIR_KEYS}

    def training(self, fold):
        values = self.arrays(fold)
        positive, negative = values['train_positive'], values['train_negative']
        A = np.zeros(self.shape, dtype=np.float32)
        A[positive[:, 0], positive[:, 1]] = 1
        mask = np.zeros(self.shape, dtype=bool)
        for pairs in (positive, negative):
            mask[pairs[:, 0], pairs[:, 1]] = True
        with np.load(self.directory / f'fold_{fold}.npz', allow_pickle=False) as source:
            allowed = source['allowed_negative_mask']
        fingerprint = array_hash(A) + ':' + array_hash(mask) + ':' + array_hash(allowed)
        return TrainingFold(fold, readonly(A), readonly(positive), readonly(negative),
                            readonly(mask), readonly(allowed), fingerprint)

    def evaluation(self, fold, part='test'):
        if part not in ('test', 'validation'):
            raise ValueError('Evaluation part must be test or validation')
        values = self.arrays(fold)
        positive, negative = values[part+'_positive'], values[part+'_negative']
        return EvaluationSet(readonly(np.vstack([positive, negative])),
                             readonly(np.r_[np.ones(len(positive)), np.zeros(len(negative))]))

    def audit(self, fold):
        values, train = self.arrays(fold), self.training(fold)
        groups = {part: set(map(tuple, np.vstack([values[part+'_positive'], values[part+'_negative']])))
                  for part in ('train', 'validation', 'test')}
        result = dict(fold=fold, train_validation_overlap=len(groups['train'] & groups['validation']),
                      train_test_overlap=len(groups['train'] & groups['test']),
                      validation_test_overlap=len(groups['validation'] & groups['test']))
        heldout = np.vstack([values[name] for name in ('validation_positive','validation_negative',
                                                      'test_positive','test_negative')])
        result['heldout_pairs_in_train_mask'] = int(train.train_mask[heldout[:,0],heldout[:,1]].sum())
        heldout_positive = np.vstack([values['validation_positive'],values['test_positive']])
        result['heldout_positives_in_A_train'] = int(train.A_train[heldout_positive[:,0],heldout_positive[:,1]].sum())
        result['candidate_negative_overlap'] = len(set(map(tuple,values['train_negative'])) & set(map(tuple,self.candidates)))
        result['heldout_pairs_in_allowed_negative_pool'] = int(train.allowed_negative_mask[heldout[:,0],heldout[:,1]].sum())
        for name,value in result.items():
            if name != 'fold' and value != 0:
                raise AssertionError(f'Isolation violation: {result}')
        scenario = self.manifest['settings']['scenario']
        test_positive = values['test_positive']
        if scenario in ('cold_circ','cold_both'):
            heldout_rows = np.unique(test_positive[:,0])
            if train.A_train[heldout_rows].any():
                raise AssertionError('Held-out circRNA has training associations')
        if scenario in ('cold_drug','cold_both'):
            heldout_columns = np.unique(test_positive[:,1])
            if train.A_train[:,heldout_columns].any():
                raise AssertionError('Held-out drug has training associations')
        result.update({name+'_count':len(values[name]) for name in PAIR_KEYS})
        result['training_fingerprint'] = train.fingerprint
        return result


def prepare_protocol(directory, association_path, candidates, *, folds=5, split_seed=114514,
                     scenario='edge', train_fraction=1.0, train_negative_ratio=1.0,
                     test_negative_ratio=1.0, validation_fraction=0.0):
    """Create once, or verify and reuse. Never silently overwrite a different protocol.

    cold_both uses folds*folds row/column block combinations, covering every known
    positive exactly once. Training excludes the union of held-out rows/columns.
    """
    directory = Path(directory).resolve()
    table = pd.read_csv(association_path,index_col=0)
    truth = table.to_numpy(dtype=np.float32)
    if not table.index.is_unique or not table.columns.is_unique:
        raise ValueError('Association identifiers must be unique')
    if not np.isfinite(truth).all() or not np.isin(truth,[0,1]).all():
        raise ValueError('Association labels must be finite binary values')
    if scenario not in ('edge','cold_circ','cold_drug','cold_both'):
        raise ValueError('Unknown experiment scenario')
    if folds < 2 or folds > min(truth.shape):
        raise ValueError('Invalid fold count')
    if not 0 < train_fraction <= 1 or not 0 <= validation_fraction < 1:
        raise ValueError('Invalid training or validation fraction')
    if (not np.isfinite(train_negative_ratio) or not np.isfinite(test_negative_ratio)
            or train_negative_ratio <= 0 or test_negative_ratio <= 0):
        raise ValueError('Negative sampling ratios must be positive')
    candidates = np.asarray(candidates,dtype=np.int64).reshape(-1,2)
    candidates = np.unique(candidates,axis=0)
    if len(candidates) and ((candidates<0).any() or (candidates[:,0]>=truth.shape[0]).any()
                            or (candidates[:,1]>=truth.shape[1]).any()):
        raise ValueError('Candidate index out of range')
    settings=dict(folds=folds,split_seed=split_seed,scenario=scenario,
                  train_fraction=train_fraction,train_negative_ratio=train_negative_ratio,
                  test_negative_ratio=test_negative_ratio,validation_fraction=validation_fraction)
    identity=dict(association_sha256=file_hash(association_path),candidate_hash=array_hash(candidates),settings=settings)
    if (directory/'manifest.json').exists():
        existing=SharedProtocol(directory)
        if existing.manifest['identity'] != identity:
            raise ValueError('Protocol settings/data changed: choose a new --protocol_dir')
        return existing
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Protocol directory is not empty; refusing to overwrite')
    directory.mkdir(parents=True,exist_ok=True)
    positives=np.argwhere(truth==1)
    if len(positives)<folds:
        raise ValueError('Too few positive associations')
    reserved=np.zeros_like(truth,dtype=bool)
    reserved[candidates[:,0],candidates[:,1]]=True
    unknown=(truth==0)&~reserved
    rng=np.random.RandomState(split_seed)
    rng.shuffle(positives)
    positive_folds=np.array_split(positives,folds)
    row_groups=np.array_split(np.random.RandomState(split_seed+31).permutation(truth.shape[0]),folds)
    col_groups=np.array_split(np.random.RandomState(split_seed+37).permutation(truth.shape[1]),folds)
    fit_count=folds*folds if scenario=='cold_both' else folds
    frozen=[]
    used_test_neg=set()
    for fold in range(fit_count):
        row_hold=np.zeros(truth.shape[0],dtype=bool)
        col_hold=np.zeros(truth.shape[1],dtype=bool)
        if scenario in ('cold_circ','cold_both'):
            row_hold[row_groups[fold//folds if scenario=='cold_both' else fold]]=True
        if scenario in ('cold_drug','cold_both'):
            col_hold[col_groups[fold%folds if scenario=='cold_both' else fold]]=True
        if scenario=='edge':
            test_pos=positive_folds[fold]
            eligible_pos=np.concatenate([value for i,value in enumerate(positive_folds) if i!=fold])
            test_region=np.ones_like(truth,dtype=bool)
            train_region=test_region.copy()
        else:
            if scenario=='cold_circ': test_region=np.broadcast_to(row_hold[:,None],truth.shape)
            elif scenario=='cold_drug': test_region=np.broadcast_to(col_hold[None,:],truth.shape)
            else: test_region=row_hold[:,None]&col_hold[None,:]
            test_pos=np.argwhere((truth==1)&test_region)
            train_region=~(row_hold[:,None]|col_hold[None,:])
            eligible_pos=np.argwhere((truth==1)&train_region)
        if not len(test_pos) or len(eligible_pos)<2:
            raise ValueError(f'Fold {fold} is empty/too small; change split seed or fold count')
        # Test sampling independent of training sparsity, training ratio, and validation size.
        test_pool=np.argwhere(unknown&test_region)
        if scenario=='edge':
            test_pool=np.array([pair for pair in test_pool if tuple(pair) not in used_test_neg],dtype=np.int64).reshape(-1,2)
        test_rng=np.random.RandomState(split_seed+100+fold)
        test_rng.shuffle(test_pool)
        n_test=max(1,int(round(len(test_pos)*test_negative_ratio)))
        if n_test>len(test_pool): raise ValueError('Insufficient unknown test pool')
        test_neg=test_pool[:n_test]
        used_test_neg.update(map(tuple,test_neg))
        train_rng=np.random.RandomState(split_seed+10000+fold)
        train_rng.shuffle(eligible_pos)
        n_val=max(1,int(round(len(eligible_pos)*validation_fraction))) if validation_fraction else 0
        val_pos=eligible_pos[:n_val]
        fitting_pos=eligible_pos[n_val:]
        if not len(fitting_pos):
            raise ValueError('Internal validation leaves no fitting positives; reduce validation_fraction')
        n_keep=max(1,int(np.floor(len(fitting_pos)*train_fraction)))
        train_pos=fitting_pos[:n_keep]
        negative_pool_mask=unknown&train_region
        negative_pool_mask=negative_pool_mask.copy()
        negative_pool_mask[test_neg[:,0],test_neg[:,1]]=False
        negative_pool=np.argwhere(negative_pool_mask)
        train_rng.shuffle(negative_pool)
        n_val_neg=max(1,int(round(len(val_pos)*train_negative_ratio))) if len(val_pos) else 0
        n_train_neg=max(1,int(round(len(train_pos)*train_negative_ratio)))
        if n_val_neg+n_train_neg>len(negative_pool): raise ValueError('Insufficient unknown training pool')
        val_neg=negative_pool[:n_val_neg]
        train_neg=negative_pool[n_val_neg:n_val_neg+n_train_neg]
        allowed_negative_mask=negative_pool_mask.copy()
        allowed_negative_mask[val_neg[:,0],val_neg[:,1]]=False
        frozen.append(dict(train_positive=train_pos,train_negative=train_neg,
                           validation_positive=val_pos,validation_negative=val_neg,
                           test_positive=test_pos,test_negative=test_neg,
                           allowed_negative_mask=allowed_negative_mask))
    union=np.vstack([value['test_positive'] for value in frozen])
    if len(union)!=len(positives) or len(set(map(tuple,union)))!=len(positives):
        raise AssertionError('Known-positive OOF coverage must be exactly once')
    artifacts={}
    np.save(directory/'candidates.npy',candidates)
    artifacts['candidates.npy']=file_hash(directory/'candidates.npy')
    for fold,values in enumerate(frozen):
        np.savez_compressed(directory/f'fold_{fold}.npz',**values)
        artifacts[f'fold_{fold}.npz']=file_hash(directory/f'fold_{fold}.npz')
    manifest=dict(schema=SCHEMA,identity=identity,settings=settings,shape=list(truth.shape),
                  fit_count=fit_count,circ_ids=list(map(str,table.index)),drug_ids=list(map(str,table.columns)),
                  artifacts=artifacts,index_base=0,known_positive_count=len(positives),
                  classification_threshold=0.5,hyperparameters='model-specific; not standardized',
                  protocol_notes=['Unknowns are presumed negatives, not confirmed negatives.',
                                  'Cold-both: Cartesian row/column test blocks, union excluded from training.',
                                  'No inner validation by default; fixed epochs and threshold.',
                                  'Training fractions use nested positive subsets; test sampling is fixed.'])
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    protocol=SharedProtocol(directory)
    for fold in range(len(protocol)): protocol.audit(fold)
    return protocol

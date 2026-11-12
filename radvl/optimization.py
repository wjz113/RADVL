"""RADVL optimization."""
from contextlib import contextmanager
import math
import random
import sys

import numpy as np
import torch


def reset_training_seed(seed):
    if not 0 <= seed < 2**32:
        raise ValueError('seed must be in [0, 2**32)')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    # DGL baselines should import DGL before calling this helper.
    dgl=sys.modules.get('dgl')
    if dgl is not None:
        dgl.seed(seed)
        if hasattr(dgl,'random'): dgl.random.seed(seed)


def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None: torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def preserved_rng():
    # The original model generates unused latent samples even in eval mode.
    # Monitoring must not change the RNG stream used by subsequent training.
    state=capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


class AucEarlyStopping:
    def __init__(self,patience=20,min_delta=0.0):
        if patience<1 or not math.isfinite(min_delta) or min_delta<0:
            raise ValueError('Invalid early-stopping configuration')
        self.patience=patience
        self.min_delta=min_delta
        self.best_auc=-math.inf
        self.best_epoch=0
        self.patience_reference=-math.inf
        self.bad_epochs=0

    def update(self,epoch,auc):
        if not math.isfinite(auc):
            raise ValueError('Validation AUC is nonfinite; refusing invalid model selection')
        is_best=auc>self.best_auc
        if is_best:
            self.best_auc=float(auc)
            self.best_epoch=epoch
        significant=auc>self.patience_reference+self.min_delta
        if significant:
            self.patience_reference=float(auc)
            self.bad_epochs=0
        else:
            self.bad_epochs+=1
        return dict(is_best=is_best,patience_reset=significant,
                    should_stop=self.bad_epochs>=self.patience)


class EpochProgress:
    def __init__(self,fold,folds,total):
        self.fold,self.folds,self.total=fold,folds,total
        self.bar=None
        try:
            from tqdm import tqdm
            self.bar=tqdm(total=total,desc=f'Fold {fold+1}/{folds}',unit='epoch',dynamic_ncols=True)
        except ImportError:
            pass

    def update(self,epoch,loss,auc,aupr,best,bad,patience):
        if self.bar is not None:
            self.bar.set_postfix(loss=f'{loss:.3f}',val_auc=f'{auc:.5f}',val_aupr=f'{aupr:.5f}',
                                 best=f'{best:.5f}',wait=f'{bad}/{patience}',refresh=False)
            self.bar.update(1)
        else:
            width=24
            done=int(width*epoch/self.total)
            print(f'Fold {self.fold+1}/{self.folds} [{"#"*done}{"-"*(width-done)}] '
                  f'{epoch}/{self.total} loss={loss:.4f} val_AUC={auc:.6f} '
                  f'val_AUPR={aupr:.6f} best_AUC={best:.6f} wait={bad}/{patience}',flush=True)

    def close(self):
        if self.bar is not None: self.bar.close()

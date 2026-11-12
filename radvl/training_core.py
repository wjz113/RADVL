"""RADVL training core."""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from radvl.settings_model import parameter_parser
from radvl.data_protocol import prepare_protocol
from radvl.features import load_external_inputs, compute_features, label_invariance_audit, build_graphs, forward
from radvl.protocol_adapter import ranking_metrics
from radvl.recorder_model import RADVLRecorder
from radvl.recorder_metadata import sync_device
from radvl.recorder_base import append_csv, METRICS
from radvl.model_core import RADVL
from radvl.optimization import reset_training_seed, preserved_rng, capture_rng, restore_rng, AucEarlyStopping, EpochProgress
from radvl.objectives import radvl_objective


def initial_state_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cpu_snapshot(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {key:cpu_snapshot(item) for key,item in value.items()}
    if isinstance(value,list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value,tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def fit(model,optimizer,training,validation,graphs,args,recorder):
    """Train on TrainingFold; monitor only internal validation; no test labels."""
    if len(validation.labels)==0 or len(np.unique(validation.labels))!=2:
        raise ValueError('AUC early stopping requires positive and negative validation pairs')
    device=graphs['A'].device
    positive_pairs=torch.tensor(training.positive_pairs.copy(),dtype=torch.long,device=device)
    negative_pairs=torch.tensor(training.negative_pairs.copy(),dtype=torch.long,device=device)
    stopping=AucEarlyStopping(args.patience,args.min_delta)
    progress=EpochProgress(training.fold,args.validation,args.epoch)
    best_state=None
    best_validation_aupr=None
    validation_seconds=0.0
    stopped_epoch=0
    sync_device(); started=time.perf_counter()
    try:
        for epoch in range(1,args.epoch+1):
            model.train()
            sync_device(); epoch_started=time.perf_counter()
            score,reconstruction,_,_,covariance,alignment=forward(model,graphs)
            total,loss_parts=radvl_objective(
                model,reconstruction,positive_pairs,negative_pairs,covariance,alignment,args)
            if not torch.isfinite(total).item():
                raise FloatingPointError('Nonfinite manuscript objective; inspect posterior and loss scales')
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            sync_device()
            recorder.epoch(training.fold,epoch,total,loss_parts,
                           model,optimizer,time.perf_counter()-epoch_started)
            # Every epoch: validation uses posterior means and no_grad/eval.
            # Preserve RNG as an additional separation from subsequent training.
            sync_device(); validation_started=time.perf_counter()
            with preserved_rng():
                validation_matrix=predict(model,graphs).detach().cpu().numpy()
            validation_scores=validation_matrix[validation.pairs[:,0],validation.pairs[:,1]]
            validation_metrics=ranking_metrics(validation.labels,validation_scores,threshold=None)
            sync_device()
            elapsed=time.perf_counter()-validation_started
            validation_seconds+=elapsed
            decision=stopping.update(epoch,validation_metrics['AUC'])
            if decision['is_best']:
                best_state=dict(model=cpu_snapshot(model.state_dict()),
                                optimizer=cpu_snapshot(optimizer.state_dict()),rng=capture_rng())
                best_validation_aupr=validation_metrics['AUPR']
            append_csv(recorder.directory/'validation_history.csv',[dict(
                fold=training.fold,epoch=epoch,validation_AUC=validation_metrics['AUC'],
                validation_AUPR=validation_metrics['AUPR'],best_AUC=stopping.best_auc,
                best_epoch=stopping.best_epoch,bad_epochs=stopping.bad_epochs,
                is_raw_best=decision['is_best'],patience_reset=decision['patience_reset'],
                validation_seconds=elapsed)])
            progress.update(epoch,float(total.detach().item()),validation_metrics['AUC'],
                            validation_metrics['AUPR'],stopping.best_auc,stopping.bad_epochs,args.patience)
            stopped_epoch=epoch
            if decision['should_stop']:
                break
    finally:
        progress.close()
    if best_state is None:
        raise RuntimeError('No valid validation checkpoint was selected')
    model.load_state_dict(best_state['model'],strict=True)
    optimizer.load_state_dict(best_state['optimizer'])
    restore_rng(best_state['rng'])
    model.eval()
    sync_device()
    summary=dict(fold=training.fold,training_seed=args.seed,best_epoch=stopping.best_epoch,
                 stopped_epoch=stopped_epoch,max_epochs=args.epoch,
                 best_validation_AUC=stopping.best_auc,
                 validation_AUPR_at_best_AUC=best_validation_aupr,
                 patience=args.patience,min_delta=args.min_delta,
                 stop_reason='patience_exhausted' if stopping.bad_epochs>=args.patience else 'max_epochs',
                 validation_seconds_total=validation_seconds,
                 restored_model_sha256=initial_state_hash(model))
    recorder.register_selection(summary)
    print(f"Fold {training.fold+1}/{args.validation}: stopped at epoch {stopped_epoch}, "
          f"restored epoch {stopping.best_epoch}, validation AUC={stopping.best_auc:.6f}",flush=True)
    return time.perf_counter()-started

@torch.no_grad()
def predict(model,graphs):
    model.eval()
    return forward(model,graphs)[0]


def main(args):
    table,circ_static,drug_static,candidates=load_external_inputs(args)
    protocol=prepare_protocol(args.protocol_dir,Path(args.data_path)/'association.csv',candidates,
        folds=args.validation,split_seed=args.split_seed,scenario=args.scenario,
        train_fraction=args.train_fraction,train_negative_ratio=args.train_negative_ratio,
        test_negative_ratio=args.test_negative_ratio,validation_fraction=args.validation_fraction)
    print('Frozen protocol:',protocol.directory,'fits:',len(protocol),flush=True)
    if args.prepare_only:
        print('Protocol created/verified. No training requested.',flush=True)
        return
    recorded_args=copy.copy(args)
    recorded_args.validation=len(protocol)
    dataset=dict(md=[],md_true=torch.tensor(table.values,dtype=torch.float32),association=table,
                 candidates=protocol.candidates,folds=[None]*len(protocol))
    for fold in range(len(protocol)):
        values=protocol.arrays(fold)
        dataset['md'].append(dict(train=[torch.from_numpy(values['train_positive']),torch.from_numpy(values['train_negative'])],
            validation=[torch.from_numpy(values['validation_positive']),torch.from_numpy(values['validation_negative'])],
            test=[torch.from_numpy(values['test_positive']),torch.from_numpy(values['test_negative'])]))
    timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory=Path('Results_radvl_aligned')/args.scenario/timestamp
    recorder=RADVLRecorder(directory,recorded_args,args.neighbors,dataset,protocol,args)
    print('Results:',directory.resolve(),flush=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold in range(len(protocol)):
        training=protocol.training(fold)
        reset_training_seed(args.seed)
        features=compute_features(training,circ_static,drug_static)
        invariant=label_invariance_audit(training,circ_static,drug_static,
                                       protocol.evaluation(fold,'test'),features)
        recorder.write_json(f'label_invariance_{fold}.json',invariant)
        recorder.fold_inputs(fold,features)
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        graphs=build_graphs(features,args.neighbors,device,args.clusters)
        recorder.graph_inputs(fold,graphs)
        fold_seed=args.seed
        reset_training_seed(fold_seed)
        model=RADVL(n_circ=table.shape[0],n_drug=table.shape[1],args=args).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
        if optimizer.state: raise AssertionError('Optimizer was not freshly initialized')
        append_csv(directory/'fold_initialization.csv',[dict(fold=fold,training_seed=fold_seed,
            initial_model_sha256=initial_state_hash(model),optimizer_state_entries=0,
            training_fingerprint=training.fingerprint,cache_policy='fresh graphs, model, optimizer')])
        validation=protocol.evaluation(fold,'validation')
        train_seconds=fit(model,optimizer,training,validation,graphs,recorded_args,recorder)
        sync_device(); inference_started=time.perf_counter()
        with preserved_rng():
            scores=predict(model,graphs)
        sync_device(); inference_seconds=time.perf_counter()-inference_started
        # Test labels are consumed only here, after optimization has finished.
        test=protocol.evaluation(fold,'test')
        scores_numpy=scores.detach().cpu().numpy()
        predictions=scores_numpy[test.pairs[:,0],test.pairs[:,1]]
        values=ranking_metrics(test.labels,predictions,threshold=0.5)
        metrics=np.array([[values[name] for name in METRICS]])
        np.save(directory/f'true_{fold}.npy',test.labels)
        np.save(directory/f'pred_{fold}.npy',predictions)
        split=dataset['md'][fold]
        recording_view=(features['ID'],features['IM'],split['train'],split['test'],
                        features['A'],dataset['md_true'],features['A'],features['reconstruction_mask'])
        recorder.performance(fold,model,train_seconds,inference_seconds)
        recorder.save_fold(fold,model,optimizer,recorded_args,recording_view,metrics)
        print(f"Fold {fold+1}/{len(protocol)} TEST (best validation-AUC checkpoint, seed {fold_seed}): "
              + ' | '.join(f'{name}={values[name]:.6f}' for name in METRICS),flush=True)
        del scores,model,optimizer,graphs,features
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    recorder.finish()
    summary=np.array([[row[name] for name in METRICS] for row in recorder.fold_metrics])
    print('TEST MEAN: '+' | '.join(f'{name}={summary[:,i].mean():.6f}' for i,name in enumerate(METRICS)),flush=True)
    print('TEST STD (population): '+' | '.join(f'{name}={summary[:,i].std():.6f}' for i,name in enumerate(METRICS)),flush=True)
    print('Results saved:',directory.resolve(),flush=True)


if __name__=='__main__':
    main(parameter_parser())



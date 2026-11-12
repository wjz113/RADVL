"""RADVL protocol metrics."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc,confusion_matrix

from radvl.data_protocol import SharedProtocol, file_hash
from radvl.optimization import reset_training_seed


def ranking_metrics(labels,scores,threshold=None):
    labels=np.asarray(labels).reshape(-1)
    scores=np.asarray(scores).reshape(-1)
    if len(labels)!=len(scores) or not np.isfinite(scores).all():
        raise ValueError('Prediction count or numerical validity failed')
    precision,recall,_=precision_recall_curve(labels,scores)
    result=dict(AUC=float(roc_auc_score(labels,scores)),AUPR=float(auc(recall,precision)))
    if threshold is not None:
        tn,fp,fn,tp=confusion_matrix(labels,scores>=threshold,labels=[0,1]).ravel()
        result.update(F1=float(2*tp/max(2*tp+fp+fn,1)),Accuracy=float((tp+tn)/len(labels)),
                      Recall=float(tp/max(tp+fn,1)),Specificity=float(tn/max(tn+fp,1)),
                      Precision=float(tp/max(tp+fp,1)))
    return result


def run_baseline_protocol(protocol_dir,output_dir,model_name,native_hyperparameters,
                          model_factory,fit_fn,predict_fn,*,training_seed=1234,
                          classification_threshold=None):
    """No fitting occurs unless explicitly called.

    model_factory(hparams,fold_seed) -> a fresh model (and fresh optimizer as needed).
    fit_fn(model,training_fold,validation_set,hparams): no test labels are supplied.
    predict_fn(model,pairs) -> 1-D scores in exactly the supplied order.
    Baseline-specific static features/graphs must be built inside fit_fn from
    training_fold.A_train plus association-independent external inputs.
    """
    protocol=SharedProtocol(protocol_dir)
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=False)
    config=dict(model_name=model_name,protocol_directory=str(protocol.directory),
                protocol_manifest_sha256=file_hash(protocol.directory/'manifest.json'),
                training_seed=training_seed,classification_threshold=classification_threshold,
                native_hyperparameters=copy.deepcopy(native_hyperparameters),
                model_hyperparameters_overridden_by_protocol=False,
                seed_policy='reset the same training_seed before every fold; split_seed belongs to protocol',
                negative_sampling='frozen training pairs; dynamic sampler must use allowed_negative_mask',
                note='Callbacks must enforce train-only preprocessing; this adapter cannot inspect arbitrary baseline internals.')
    (output/'baseline_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    all_predictions=[]; metrics=[]
    for fold in range(len(protocol)):
        protocol.audit(fold)
        train=protocol.training(fold)
        validation=protocol.evaluation(fold,'validation')
        hparams=copy.deepcopy(native_hyperparameters)
        reset_training_seed(training_seed)
        model=model_factory(hparams,training_seed)
        started=time.perf_counter()
        fit_fn(model,train,validation,hparams)
        fitting_seconds=time.perf_counter()-started
        # The held-out labels become available only after fitting.
        test=protocol.evaluation(fold,'test')
        scores=np.asarray(predict_fn(model,test.pairs)).reshape(-1)
        if len(scores)!=len(test.labels): raise ValueError('Baseline prediction order/count mismatch')
        values=ranking_metrics(test.labels,scores,classification_threshold)
        metrics.append(dict(fold=fold,training_seed=training_seed,fit_seconds=fitting_seconds,**values))
        pairs=test.pairs
        rows=pd.DataFrame(dict(fold=fold,circ_index=pairs[:,0],drug_index=pairs[:,1],
                               circ_id=[protocol.circ_ids[i] for i in pairs[:,0]],
                               drug_id=[protocol.drug_ids[j] for j in pairs[:,1]],
                               label=test.labels,prediction=scores))
        rows.to_csv(output/f'predictions_fold_{fold}.csv',index=False)
        all_predictions.append(rows)
        pd.DataFrame(metrics).to_csv(output/'fold_metrics.csv',index=False)
        del model
    merged=pd.concat(all_predictions,ignore_index=True)
    if merged.duplicated(['circ_index','drug_index']).any():
        raise AssertionError('Repeated OOF test pair')
    merged.to_csv(output/'oof_predictions.csv',index=False)
    merged[merged.label==1].to_csv(output/'known_edges_oof.csv',index=False)
    columns=[name for name in metrics[0] if name not in ('fold','training_seed','fit_seconds')]
    frame=pd.DataFrame(metrics)[columns]
    pd.DataFrame([dict(statistic='mean',**frame.mean().to_dict()),
                  dict(statistic='std_population',**frame.std(ddof=0).to_dict())]).to_csv(
                      output/'summary_metrics.csv',index=False)
    return output


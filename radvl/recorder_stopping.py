"""RADVL recorder stopping."""
import copy
import json
import shutil
from pathlib import Path

from radvl.recorder_fusion import FusionMLPRecorder
from radvl.recorder_base import append_csv
from radvl.data_protocol import file_hash


class EarlyStopRecorder(FusionMLPRecorder):
    def __init__(self,directory,args,k,dataset,protocol,requested_args):
        super().__init__(directory,args,k,dataset,protocol,requested_args)
        self.selections={}
        root=Path(__file__).resolve().parent
        config=json.loads((self.directory/'run_config.json').read_text(encoding='utf-8'))
        for name in ['training_early_stopping.py','settings_stopping.py',
                     'recorder_stopping.py','optimization.py','protocol_metrics.py']:
            shutil.copy2(root/name,self.directory/'source'/name)
            config['input_and_source_sha256']['source/'+name]=file_hash(root/name)
        config.update(early_stopping=True,selection_metric='internal validation AUC only',
                      patience=args.patience,min_delta=args.min_delta,
                      checkpoint_policy='raw highest validation AUC; first epoch wins exact ties',
                      patience_policy='reset when AUC exceeds patience reference by min_delta',
                      training_seed_policy='same seed reset before every fold; no fold offset',
                      training_seed=args.seed,split_seed=args.split_seed,
                      validation_rng_policy='save and restore Python/NumPy/Torch/CUDA RNG state',
                      cudnn_deterministic=True,cudnn_benchmark=False,
                      test_policy='restore validation-best model and optimizer; one final test forward per fold')
        config['notes']=[note for note in config.get('notes',[]) if
                         not any(text in note for text in ['No hyperparameter search, no early stopping',
                                  'Optional internal validation','base training seed + fit index',
                                  'Final checkpoint'])]
        config['notes'].append('Validation AUPR is recorded but never used for stopping, tie-breaking or selection.')
        self.write_json('run_config.json',config)

    def register_selection(self,summary):
        self.selections[summary['fold']]=summary
        append_csv(self.directory/'early_stopping_summary.csv',[summary])

    def save_fold(self,fold,model,optimizer,args,data,metrics):
        selection=self.selections[fold]
        selected_args=copy.copy(args)
        selected_args.max_epochs=args.epoch
        selected_args.epoch=selection['best_epoch']
        selected_args.best_epoch=selection['best_epoch']
        selected_args.stopped_epoch=selection['stopped_epoch']
        selected_args.validation_selection=selection
        # Parent stores the already-restored model and matching optimizer, not
        # the model from the stopping epoch. The checkpoint epoch is accurate.
        super().save_fold(fold,model,optimizer,selected_args,data,metrics)
        if getattr(args, 'save_checkpoints', 'on') == 'on':
            (self.directory/f'model_fold_{fold}_final.pt').replace(
                self.directory/f'model_fold_{fold}_best.pt')

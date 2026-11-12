"""RADVL recorder fusion."""
import json
import shutil
from pathlib import Path
from radvl.recorder_protocol import ProtocolRecorder
from radvl.data_protocol import file_hash
from radvl.recorder_base import append_csv


class FusionMLPRecorder(ProtocolRecorder):
    def __init__(self,directory,args,k,dataset,protocol,requested_args):
        super().__init__(directory,args,k,dataset,protocol,requested_args)
        config=json.loads((self.directory/'run_config.json').read_text(encoding='utf-8'))
        root=Path(__file__).resolve().parent
        for name in ['training_fusion.py','model_fusion.py',
                     'settings_fusion.py','recorder_fusion.py']:
            shutil.copy2(root/name,self.directory/'source'/name)
            config['input_and_source_sha256']['source/'+name]=file_hash(root/name)
        config.update(model='radvl_cross_view_fusion_pairwise_mlp',
            model_prediction='sigmoid(MLP(circ_fused elementwise_product drug_fused)); no 0.9/0.1 mixture',
            similarity_view='concatenate GIN and HGCN outputs then project to shared width',
            fusion_view_pair='similarity embedding and CoVAE posterior mean embedding',
            fusion_alignment='exp(-(squared_delta1+squared_delta2)/fusion_tau), weighted squared L2; sum both entity sides',
            prediction_loss='BCEWithLogits sum on frozen training mask, no alpha weighting',
            fusion_tau=args.fusion_tau,fusion_alignment_weight=args.fusion_alignment_weight,
            predictor_hidden_dim=args.pair_hidden_dim,pair_chunk_rows=args.pair_chunk_rows,
            diagnostic_score_meanings=dict(association='posterior-mean inner product, diagnostic only',
                fused_inner_product='fused inner product, diagnostic only; NOT MLP output',
                combined='actual final MLP probability'),
            limitations=['Original CoVAE input, mean decoding, covariance/KL and explicit L2 retained.',
                         'Original graph construction, including edge orientation, retained.',
                         'Fusion tau=100 is an initial numerical scale, not a validated optimum.',
                         'Not all manuscript equations have been reimplemented.'])
        config['notes']=[note for note in config.get('notes',[]) if 'Original architecture' not in note]
        config['notes'].append('Fusion, final predictor and primary loss changed by explicit user request.')
        self.write_json('run_config.json',config)

    def epoch(self,fold,epoch,total,losses,model,optimizer,compute_seconds):
        super().epoch(fold,epoch,total,losses,model,optimizer,compute_seconds)
        logits=model.prediction_logits.detach()
        scores=model.diagnostic_scores['combined']
        gradient=sum(parameter.grad.detach().float().square().sum().item()
                     for parameter in model.pair_predictor.parameters() if parameter.grad is not None)**0.5
        append_csv(self.directory/'predictor_diagnostics.csv',[dict(
            fold=fold,epoch=epoch,mlp_gradient_norm=gradient,
            logit_mean=logits.mean().item(),logit_std=logits.std(unbiased=False).item(),
            logit_min=logits.min().item(),logit_max=logits.max().item(),
            probability_ge_0999_fraction=(scores>=0.999).float().mean().item(),
            probability_le_0001_fraction=(scores<=0.001).float().mean().item(),
            scope='all-pair forward predictions; no all-pair label supervision')])

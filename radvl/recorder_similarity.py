"""RADVL recorder similarity."""
import json
from pathlib import Path
import shutil

import numpy as np

from radvl.recorder_model import RADVLRecorder
from radvl.recorder_base import append_csv
from radvl.data_protocol import file_hash


WEIGHTED_SOURCE_FILES = ('hypergraph_weighting.py',
                         'features_similarity.py',
                         'model_similarity.py',
                         'settings_similarity.py',
                         'recorder_similarity.py',
                         'training_weighted.py',
                         'test_similarity_fusion.py')


class RADVLWeightedRecorder(RADVLRecorder):
    def __init__(self, directory, args, k, dataset, protocol, requested_args):
        super().__init__(directory, args, k, dataset, protocol, requested_args)
        root = Path(__file__).resolve().parent
        config = json.loads((self.directory / 'run_config.json').read_text(encoding='utf-8'))
        for name in WEIGHTED_SOURCE_FILES:
            shutil.copy2(root / name, self.directory / 'source' / name)
            config['input_and_source_sha256']['source/' + name] = file_hash(root / name)
        config.update(
            model='radvl_aligned_covariance_CoVAE_weighted_graphs_cross_view_MLP',
            modified_components=['GIN neighbour weighting', 'hyperedge weight matrix W', 'fusion alignment temperature'],
            hypergraph_policy=('K-means similarity-row clusters; unit hyperedge weights; '
                               'G = Dv^-1/2 H W De^-1 H^T Dv^-1/2'
                               if args.hyperedge_weight == 'unit' else
                               'K-means similarity-row clusters; w_e = mean intra-hyperedge similarity; '
                               'G = Dv^-1/2 H W De^-1 H^T Dv^-1/2'),
            hyperedge_weight_policy=dict(mode=args.hyperedge_weight,
                                        rescale=args.hyperedge_weight_rescale == 'on',
                                        definition=('W = I (Eq. 18 as currently written)' if args.hyperedge_weight == 'unit'
                                                    else 'w_e = mean over ordered member pairs of the fused similarity'),
                                        note='the manuscript denotes W = diag(w_1..w_n) but never defines w_e'),
            gin_neighbor_aggregation=('unweighted sum of Eq. (15)' if args.gin_weighting == 'uniform' else
                                      f"fused similarity-weighted sum, normalization={args.gin_weight_normalization}"),
            gin_weighting=args.gin_weighting, gin_weight_normalization=args.gin_weight_normalization,
            fusion_tau_policy=dict(mode=args.fusion_tau_mode, tau=args.fusion_tau,
                                   tau_scale=args.fusion_tau_scale,
                                   note=('constant tau as in the current release' if args.fusion_tau_mode == 'fixed'
                                         else 'tau = fusion_tau_scale * mean(D) per forward pass; D is O(1e4) at '
                                              'initialisation and O(1e3) after convergence, so a constant tau=100 '
                                              'sends exp(-D/tau) below 1e-8 for over 90% of the nodes')),
            fusion_tau=args.fusion_tau, fusion_tau_mode=args.fusion_tau_mode,
            fusion_tau_scale=args.fusion_tau_scale,
            unchanged_components='model topology, CoVAE, objective, protocol, early stopping, threshold, metrics',
            limitations=list(config.get('limitations', []))
                        + ['The added switches default to the previous behaviour; only non-default values change the run.',
                           'This variant has not been trained by the editing task that created it.'])
        config['notes'].append('GIN edge weights, the hyperedge weight matrix and the alignment temperature are now '
                               'explicit policies recorded in run_config.json.')
        self.write_json('run_config.json', config)

    def graph_inputs(self, fold, graphs):
        """Parent arrays and audit, plus honest names for the weighted operator and its weights."""
        super().graph_inputs(fold, graphs)
        arrays = dict(circ_hypergraph_operator=graphs['circ_hypergraph_operator'].detach().cpu().numpy(),
                      drug_hypergraph_operator=graphs['drug_hypergraph_operator'].detach().cpu().numpy(),
                      circ_hyperedge_weights=graphs['circ_hyperedge_weights'].detach().cpu().numpy(),
                      drug_hyperedge_weights=graphs['drug_hyperedge_weights'].detach().cpu().numpy(),
                      circ_kmeans_incidence=graphs['circ_incidence'].detach().cpu().numpy(),
                      drug_kmeans_incidence=graphs['drug_incidence'].detach().cpu().numpy())
        np.savez_compressed(self.directory / f'weighted_graph_{fold}.npz', **arrays)
        for side in ['circ', 'drug']:
            diagnostics = dict(graphs[side + '_hyperedge_weight_diagnostics'])
            weights = graphs[side + '_hyperedge_weights'].detach().cpu().numpy()
            operator = graphs[side + '_hypergraph_operator'].detach().cpu().numpy()
            if not np.isfinite(operator).all():
                raise AssertionError('Nonfinite hypergraph propagation operator')
            if (weights <= 0).any():
                raise AssertionError('Hyperedge weights must be strictly positive')
            append_csv(self.directory / 'hyperedge_weight_audit.csv', [dict(
                fold=fold, side=side, request=graphs['hyperedge_weight'],
                rescaled=graphs['hyperedge_weight_rescale'],
                hyperedge_count=diagnostics['hyperedge_count'],
                weight_min=diagnostics['weight_min'], weight_max=diagnostics['weight_max'],
                weight_mean=diagnostics['weight_mean'],
                floored_hyperedge_count=diagnostics['floored_hyperedge_count'],
                raw_weight_mean_before_floor=diagnostics.get('raw_weight_mean_before_floor', float('nan')),
                operator_diagonal_min=float(np.diag(operator).min()),
                operator_trace=float(np.trace(operator)),
                gin_weighting=getattr(self.args, 'gin_weighting', None),
                fusion_tau_mode=getattr(self.args, 'fusion_tau_mode', None))])

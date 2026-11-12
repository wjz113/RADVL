"""RADVL recorder model."""
import json
from pathlib import Path
import shutil

import numpy as np

from radvl.recorder_stopping import EarlyStopRecorder
from radvl.recorder_base import append_csv
from radvl.data_protocol import array_hash, file_hash


class RADVLRecorder(EarlyStopRecorder):
    def __init__(self, directory, args, k, dataset, protocol, requested_args):
        super().__init__(directory, args, k, dataset, protocol, requested_args)
        root = Path(__file__).resolve().parent
        config = json.loads((self.directory / 'run_config.json').read_text(encoding='utf-8'))
        for name in ['training_core.py', 'settings_model.py',
                     'model_core.py', 'features.py',
                     'objectives.py', 'recorder_model.py']:
            shutil.copy2(root / name, self.directory / 'source' / name)
            config['input_and_source_sha256']['source/' + name] = file_hash(root / name)
        config.update(
            model='radvl_aligned_covariance_CoVAE_cross_view_MLP',
            model_prediction='sigmoid(MLP(elementwise product of the two fused entity embeddings))',
            association_view_input='circRNA A_train rows; drug A_train columns; no similarity concatenation',
            posterior_parameterization='mean and log variance; log_std aliases are one half of log variance',
            training_latent_policy='reparameterized samples used for reconstruction and association-view fusion',
            evaluation_latent_policy='posterior means; no latent noise during validation/test',
            covariance='Eq30: sum of both within-side squared off-diagonal covariances / (2*d*(d-1))',
            reconstruction='Eq27: eta_pos*mean positive BCEWithLogits + eta_neg*mean negative BCEWithLogits; training pairs only',
            prediction_loss='Eq44: BCEWithLogits summed over training positive and negative pairs only',
            kl_policy='Eq31: standard Gaussian KL, sum over latent coordinates, average separately over each entity side',
            total_objective='prediction + lambda_av*(reconstruction + KL_circ + KL_drug + lambda_cov*covariance) + lambda_align*alignment',
            explicit_L2_loss=False, optimizer=dict(name='AdamW', lr=args.learning_rate, weight_decay=args.weight_decay),
            lambda_cov=args.lambda_cov, lambda_av=args.lambda_av, lambda_align=args.lambda_align,
            eta_pos=args.eta_pos, eta_neg=args.eta_neg,
            loss_weights=dict(covariance=args.lambda_cov, association_view=args.lambda_av,
                              alignment=args.lambda_align, eta_pos=args.eta_pos, eta_neg=args.eta_neg),
            similarity_view='GIN and HGCN receive fold-specific fused similarity rows only',
            graph_policy='symmetric union of row-wise top-k positive similarities; unique bidirectional edges; no GIN self loops',
            gin_neighbor_aggregation='unweighted sum; edge_attr stores similarities for provenance only',
            hypergraph_policy='K-means similarity-row clusters; unit hyperedge weights; normalized incidence propagation',
            kmeans_seed=0, kmeans_n_init=10, kmeans_clusters_requested=args.clusters,
            neighbors=args.neighbors, hidden_list=[args.hypergraph_width], latent_dim=args.latent_dim,
            num_hidden1=args.vae_hidden_dim, gin_layers=args.gin_layers, hgcn_layers_actual=args.hgcn_layers,
            architecture=dict(hidden_dim=args.hid_dim, latent_dim=args.latent_dim,
                              vae_hidden_dim=args.vae_hidden_dim, hypergraph_width=args.hypergraph_width,
                              gin_layers=args.gin_layers, hgcn_layers=args.hgcn_layers,
                              neighbors=args.neighbors, clusters=args.clusters,
                              pair_hidden_dim=args.pair_hidden_dim, dropout=args.dropout),
            hyperparameter_search=False,
            static_provenance_status='user-confirmed manuscript sequence/structure similarity results; existing files reused unchanged',
            limitations=['New implementation has not been executed or trained by this editing task.',
                         'Uses existing FastKAN/HGCN primitives; this change does not claim to rederive every encoder equation.',
                         'Default model settings and fusion_tau are starting configurations, not selected optima.'],
            notes=['Separate new entry/model/graph/objective files; old files and results retained.',
                   'Validation AUC alone selects the checkpoint; validation AUPR is descriptive only.',
                   'Test threshold is fixed at 0.5; no test-derived threshold or checkpoint choice.',
                   'Every fold resets the same requested training seed; no fold offset.',
                   'No dynamic cross-covariance temperature co_tau is used in this objective.',
                   'AdamW decay remains an optimizer setting, not an extra explicit L2 term in Eq45.',
                   'Static inputs are reused according to user confirmation, without recomputing their source similarities.'])
        config.pop('num_layers_argument', None)
        self.write_json('run_config.json', config)

    def graph_inputs(self, fold, graphs):
        tensors = dict(association_input=graphs['A'],
                       circ_features=graphs['circ_features'], drug_features=graphs['drug_features'],
                       circ_kmeans_operator=graphs['circ_kmeans'], drug_kmeans_operator=graphs['drug_kmeans'],
                       circ_hypergraph_incidence=graphs['circ_incidence'], drug_hypergraph_incidence=graphs['drug_incidence'],
                       circ_cluster_labels=graphs['circ_cluster_labels'], drug_cluster_labels=graphs['drug_cluster_labels'],
                       circ_graph_x=graphs['circ_graph'].x, drug_graph_x=graphs['drug_graph'].x,
                       circ_edge_index=graphs['circ_graph'].edge_index, drug_edge_index=graphs['drug_graph'].edge_index,
                       circ_edge_attr=graphs['circ_graph'].edge_attr, drug_edge_attr=graphs['drug_graph'].edge_attr)
        arrays = {name: value.detach().cpu().numpy() for name, value in tensors.items()}
        np.savez_compressed(self.directory / f'graph_inputs_{fold}.npz', **arrays)
        training = self.protocol.training(fold)
        append_csv(self.directory / 'feature_lineage.csv', [dict(
            fold=fold, artifact=name, sha256=array_hash(value), training_fingerprint=training.fingerprint,
            source='this fold training associations and existing static sequence/structure inputs')
            for name, value in arrays.items()])
        for side in ['circ', 'drug']:
            graph = graphs[side + '_graph']
            edges = graph.edge_index.detach().cpu().numpy().T
            weights = graph.edge_attr.detach().cpu().numpy()
            pairs = {tuple(pair): float(weight) for pair, weight in zip(edges, weights)}
            loops = sum(i == j for i, j in pairs)
            missing = sum((j, i) not in pairs for i, j in pairs)
            inconsistent = sum((j, i) in pairs and not np.isclose(weight, pairs[(j, i)])
                               for (i, j), weight in pairs.items())
            if loops or missing or inconsistent or len(pairs) != len(edges):
                raise AssertionError('Invalid undirected loop-free similarity graph')
            append_csv(self.directory / 'graph_audit.csv', [dict(
                fold=fold, side=side, directed_edge_count=len(edges), self_loop_count=loops,
                missing_reverse_count=missing, reverse_weight_mismatch_count=inconsistent,
                duplicate_edge_count=len(edges) - len(pairs),
                hyperedge_count=graphs[side + '_incidence'].shape[1],
                hgcn_uses='saved K-means incidence propagation operator')])

    def save_fold(self, fold, model, optimizer, args, data, metrics):
        super().save_fold(fold, model, optimizer, args, data, metrics)
        np.savez_compressed(self.directory / f'posterior_{fold}.npz',
                            circ_mu=model.z_node_mean.detach().cpu().numpy(),
                            circ_log_variance=model.z_node_log_var.detach().cpu().numpy(),
                            drug_mu=model.z_edge_mean.detach().cpu().numpy(),
                            drug_log_variance=model.z_edge_log_var.detach().cpu().numpy())

"""RADVL model similarity."""
import torch
from torch import nn

from radvl.model_components import FastKANLayer, Innovative_HGCN
from radvl.objectives import covariance_penalty


GIN_WEIGHTINGS = ('uniform', 'similarity')
GIN_NORMALIZATIONS = ('mass', 'mean')
FUSION_TAU_MODES = ('fixed', 'adaptive')


class WeightedGINLayer(nn.Module):
    def __init__(self, width, dropout, weighting='uniform', normalization='mass'):
        super().__init__()
        if weighting not in GIN_WEIGHTINGS:
            raise ValueError(f'Unknown GIN weighting: {weighting}')
        if normalization not in GIN_NORMALIZATIONS:
            raise ValueError(f'Unknown GIN weight normalization: {normalization}')
        self.weighting = weighting
        self.normalization = normalization
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            FastKANLayer(input_dim=width, output_dim=width, num_grids=8,
                         grid_min=-2, grid_max=2, use_base_update=True),
            nn.Dropout(dropout), nn.ReLU(), nn.Linear(width, width))

    def aggregate(self, x, edge_index, edge_attr):
        receiver, neighbor = edge_index
        aggregate = torch.zeros_like(x)
        if self.weighting == 'uniform':
            aggregate.index_add_(0, receiver, x[neighbor])
            return aggregate
        if edge_attr is None:
            raise ValueError('Similarity weighting requires edge_attr on the similarity graph')
        weights = edge_attr.to(x.dtype)
        if (weights <= 0).any():
            raise ValueError('Similarity edge weights must be strictly positive')
        degree = torch.zeros(x.shape[0], dtype=weights.dtype, device=weights.device)
        degree.index_add_(0, receiver, weights)
        neighbours = torch.zeros_like(degree)
        neighbours.index_add_(0, receiver, torch.ones_like(weights))
        scale = torch.ones_like(degree)
        present = degree > 0
        if self.normalization == 'mass':
            # sum_j w_ij * scale_i = number of neighbours of i, as in the unweighted sum.
            scale[present] = neighbours[present] / degree[present]
        else:
            scale[present] = 1.0 / degree[present]
        aggregate.index_add_(0, receiver, x[neighbor] * (weights * scale[receiver]).unsqueeze(-1))
        return aggregate

    def forward(self, x, edge_index, edge_attr=None):
        # The graph builder removes loops; self contribution occurs exactly once.
        return self.mlp((1 + self.eps) * x + self.aggregate(x, edge_index, edge_attr))


class WeightedGINEncoder(nn.Module):
    def __init__(self, input_dim, width, layers, dropout, weighting='uniform', normalization='mass'):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, width)
        self.layers = nn.ModuleList([WeightedGINLayer(width, dropout, weighting, normalization)
                                     for _ in range(layers)])

    def forward(self, x, edge_index, edge_attr=None):
        x = self.fc_in(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x


class RADVLFusion(nn.Module):
    """Eqs. (32)-(42) with a selectable alignment-temperature policy."""

    def __init__(self, dimension, tau, tau_mode='fixed', tau_scale=1.0):
        super().__init__()
        if tau_mode not in FUSION_TAU_MODES:
            raise ValueError(f'Unknown fusion tau mode: {tau_mode}')
        self.tau = tau
        self.scale = dimension ** -0.5
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.gate_1 = nn.Linear(3 * dimension, dimension)
        self.gate_2 = nn.Linear(3 * dimension, dimension)
        self.tau_mode = tau_mode
        self.tau_scale = tau_scale

    def context(self, first, second):
        attention = torch.softmax((self.query(first) @ self.key(second).T) * self.scale, dim=-1)
        return attention @ self.value(second)

    def forward(self, similarity, association):
        context_1 = self.context(similarity, association)
        context_2 = self.context(association, similarity)
        delta_1 = similarity - context_1
        delta_2 = association - context_2
        gate_1 = torch.sigmoid(self.gate_1(torch.cat([delta_1, similarity, context_1], dim=-1)))
        gate_2 = torch.sigmoid(self.gate_2(torch.cat([delta_2, association, context_2], dim=-1)))
        first = gate_1 * similarity + (1 - gate_1) * context_1
        second = gate_2 * association + (1 - gate_2) * context_2
        discrepancy = delta_1.square().sum(dim=-1) + delta_2.square().sum(dim=-1)
        if self.tau_mode == 'adaptive':
            tau = (self.tau_scale * discrepancy.mean()).detach().clamp_min(1e-12)
        else:
            tau = self.tau
        weight = torch.exp(-discrepancy / tau)
        alignment = (weight * (first - second).square().sum(dim=-1)).mean()
        fused = first + second
        with torch.no_grad():
            self.diagnostics = dict(delta_squared_mean=discrepancy.mean().item(),
                tau_effective=float(tau),
                alignment_strength_mean=weight.mean().item(), alignment_strength_min=weight.min().item(),
                alignment_strength_max=weight.max().item(),
                alignment_weight_below_1e_minus8_fraction=(weight < 1e-8).float().mean().item(),
                gate_1_mean=gate_1.mean().item(), gate_2_mean=gate_2.mean().item(),
                fused_norm_mean=fused.norm(dim=1).mean().item())
            self.diagnostic_arrays = dict(delta_1=delta_1.detach(), delta_2=delta_2.detach(),
                gate_1=gate_1.detach(), gate_2=gate_2.detach(), alignment_strength=weight.detach())
        return fused, alignment


class RADVL(nn.Module):
    def __init__(self, n_circ, n_drug, args):
        super().__init__()
        self.n_circ, self.n_drug = n_circ, n_drug
        width, hyper_width, latent = args.hid_dim, args.hypergraph_width, args.latent_dim
        gin_weighting = getattr(args, 'gin_weighting', 'uniform')
        gin_normalization = getattr(args, 'gin_weight_normalization', 'mass')
        tau_mode = getattr(args, 'fusion_tau_mode', 'fixed')
        tau_scale = getattr(args, 'fusion_tau_scale', 1.0)
        self.circ_fastkan = FastKANLayer(input_dim=n_circ, output_dim=hyper_width,
                                       num_grids=8, grid_min=-2, grid_max=2, use_base_update=True)
        self.dr_fastkan = FastKANLayer(input_dim=n_drug, output_dim=hyper_width,
                                     num_grids=8, grid_min=-2, grid_max=2, use_base_update=True)
        # A one-element hidden_list makes args.hgcn_layers the ACTUAL layer count.
        hgcn = dict(in_dim=hyper_width, hidden_list=[hyper_width], num_layers=args.hgcn_layers,
                    dropout=args.dropout, use_edge_attention=True,
                    use_feature_recalibration=True, use_jk=True)
        self.hgcn_circ = Innovative_HGCN(**hgcn)
        self.hgcn_dr = Innovative_HGCN(**hgcn)
        self.circ_gin_encoder = WeightedGINEncoder(n_circ, width, args.gin_layers, args.dropout,
                                                  gin_weighting, gin_normalization)
        self.dr_gin_encoder = WeightedGINEncoder(n_drug, width, args.gin_layers, args.dropout,
                                                gin_weighting, gin_normalization)
        self.circ_global_projection = nn.Linear(hyper_width, width)
        self.dr_global_projection = nn.Linear(hyper_width, width)
        self.circ_similarity_projection = nn.Linear(2 * width, width)
        self.dr_similarity_projection = nn.Linear(2 * width, width)
        # CircRNA encoder sees A_train rows; drug encoder sees A_train columns.
        self.node_encoders1 = nn.Sequential(nn.Linear(n_drug, args.vae_hidden_dim),
                                           nn.ReLU(), nn.Dropout(args.dropout))
        self.hyperedge_encoders1 = nn.Sequential(nn.Linear(n_circ, args.vae_hidden_dim),
                                                nn.ReLU(), nn.Dropout(args.dropout))
        self.circ_mean = nn.Linear(args.vae_hidden_dim, latent)
        self.circ_log_variance = nn.Linear(args.vae_hidden_dim, latent)
        self.drug_mean = nn.Linear(args.vae_hidden_dim, latent)
        self.drug_log_variance = nn.Linear(args.vae_hidden_dim, latent)
        self.circ_association_projection = nn.Identity() if latent == width else nn.Linear(latent, width)
        self.dr_association_projection = nn.Identity() if latent == width else nn.Linear(latent, width)
        self.circ_fusion = RADVLFusion(width, args.fusion_tau, tau_mode, tau_scale)
        self.dr_fusion = RADVLFusion(width, args.fusion_tau, tau_mode, tau_scale)
        self.pair_predictor = nn.Sequential(nn.Linear(width, args.pair_hidden_dim), nn.ReLU(),
                                            nn.Linear(args.pair_hidden_dim, 1))
        self.pair_chunk_rows = args.pair_chunk_rows

    def pair_logits(self, circ, drug):
        chunks = []
        for start in range(0, len(circ), self.pair_chunk_rows):
            products = circ[start:start + self.pair_chunk_rows, None, :] * drug[None, :, :]
            chunks.append(self.pair_predictor(products).squeeze(-1))
        return torch.cat(chunks, dim=0)

    def latent(self, mean, log_variance):
        # Train: Eqs. (24)-(25). Validation/test: deterministic posterior means.
        return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean) if self.training else mean

    def forward(self, A, circ_graph, drug_graph, circ_hypergraph, drug_hypergraph):
        if A.shape != (self.n_circ, self.n_drug):
            raise ValueError('A_train shape does not match model entity dimensions')
        circ_local = self.circ_gin_encoder(circ_graph.x, circ_graph.edge_index, circ_graph.edge_attr)
        drug_local = self.dr_gin_encoder(drug_graph.x, drug_graph.edge_index, drug_graph.edge_attr)
        circ_global = self.circ_global_projection(self.hgcn_circ(self.circ_fastkan(circ_graph.x), circ_hypergraph))
        drug_global = self.dr_global_projection(self.hgcn_dr(self.dr_fastkan(drug_graph.x), drug_hypergraph))
        circ_similarity = self.circ_similarity_projection(torch.cat([circ_local, circ_global], dim=-1))
        drug_similarity = self.dr_similarity_projection(torch.cat([drug_local, drug_global], dim=-1))
        circ_encoded, drug_encoded = self.node_encoders1(A), self.hyperedge_encoders1(A.T)
        self.z_node_mean, self.z_node_log_var = self.circ_mean(circ_encoded), self.circ_log_variance(circ_encoded)
        self.z_edge_mean, self.z_edge_log_var = self.drug_mean(drug_encoded), self.drug_log_variance(drug_encoded)
        # Log-std aliases support existing recorders without changing KL semantics.
        self.z_node_log_std, self.z_edge_log_std = 0.5 * self.z_node_log_var, 0.5 * self.z_edge_log_var
        self.z_node_sample = self.latent(self.z_node_mean, self.z_node_log_var)
        self.z_edge_sample = self.latent(self.z_edge_mean, self.z_edge_log_var)
        reconstruction_logits = self.z_node_sample @ self.z_edge_sample.T
        circ_association = self.circ_association_projection(self.z_node_sample)
        drug_association = self.dr_association_projection(self.z_edge_sample)
        circ_fused, circ_alignment = self.circ_fusion(circ_similarity, circ_association)
        drug_fused, drug_alignment = self.dr_fusion(drug_similarity, drug_association)
        alignment = circ_alignment + drug_alignment
        self.prediction_logits = self.pair_logits(circ_fused, drug_fused)
        scores = torch.sigmoid(self.prediction_logits)
        covariance = covariance_penalty(self.z_node_mean, self.z_edge_mean)
        self.diagnostic_scores = dict(
            association=torch.sigmoid(self.z_node_mean @ self.z_edge_mean.T).detach(),
            fused_inner_product=torch.sigmoid(circ_fused @ drug_fused.T).detach(), combined=scores.detach())
        self.diagnostic_embeddings = dict(circ=circ_fused.detach(), drug=drug_fused.detach())
        return scores, reconstruction_logits, drug_fused, circ_fused, covariance, alignment

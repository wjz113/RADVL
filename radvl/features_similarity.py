"""RADVL features similarity."""
import torch

from radvl.feature_inputs import load_external_inputs, compute_features, label_invariance_audit
from radvl.features import similarity_graph
from radvl.hypergraph_weighting import weighted_hypergraph, WEIGHT_MODES


def build_graphs(features, neighbors, device, clusters=9, hyperedge_weight='unit',
                 hyperedge_weight_rescale=True):
    """GIN similarity graph (unchanged) plus the possibly weighted hypergraph operator."""
    if hyperedge_weight not in WEIGHT_MODES:
        raise ValueError(f'Unknown hyperedge weight mode: {hyperedge_weight}')
    circ_similarity = features['IM'].detach().cpu().numpy()
    drug_similarity = features['ID'].detach().cpu().numpy()
    circ_operator, circ_incidence, circ_labels, circ_weights, circ_diag = weighted_hypergraph(
        circ_similarity, clusters, hyperedge_weight, hyperedge_weight_rescale)
    drug_operator, drug_incidence, drug_labels, drug_weights, drug_diag = weighted_hypergraph(
        drug_similarity, clusters, hyperedge_weight, hyperedge_weight_rescale)
    circ_operator_t = circ_operator.to(device)
    drug_operator_t = drug_operator.to(device)
    return dict(
        A=features['A'].to(device),
        circ_features=features['IM'].to(device), drug_features=features['ID'].to(device),
        circ_hypergraph_operator=circ_operator_t, drug_hypergraph_operator=drug_operator_t,
        circ_incidence=circ_incidence, drug_incidence=drug_incidence,
        circ_hyperedge_weights=circ_weights, drug_hyperedge_weights=drug_weights,
        circ_cluster_labels=circ_labels, drug_cluster_labels=drug_labels,
        circ_hyperedge_weight_diagnostics=circ_diag, drug_hyperedge_weight_diagnostics=drug_diag,
        hyperedge_weight=hyperedge_weight, hyperedge_weight_rescale=bool(hyperedge_weight_rescale),
        # Legacy aliases required by the shared training/evaluation loop.
        circ_kmeans=circ_operator_t, drug_kmeans=drug_operator_t,
        circ_graph=similarity_graph(circ_similarity, neighbors).to(device),
        drug_graph=similarity_graph(drug_similarity, neighbors).to(device))


def forward(model, graphs):
    """Same call signature as the parent; delegates to the shared forward."""
    from radvl.features import forward as shared_forward
    return shared_forward(model, graphs)

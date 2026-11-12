"""RADVL features."""
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch_geometric.data import Data

from radvl.feature_inputs import load_external_inputs, compute_features, label_invariance_audit


def similarity_graph(similarity, neighbors):
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError('Similarity must be square')
    if not np.isfinite(similarity).all() or (similarity < 0).any():
        raise ValueError('Similarity must be finite and nonnegative')
    if not np.allclose(similarity, similarity.T, rtol=1e-5, atol=1e-6):
        raise ValueError('The undirected similarity view requires symmetric similarities')
    count = len(similarity)
    if not 0 < neighbors < count:
        raise ValueError('neighbors must be between 1 and node_count-1')
    weights = (similarity + similarity.T) / 2
    ranking = weights.copy()
    np.fill_diagonal(ranking, -np.inf)
    selected = np.argsort(-ranking, axis=1, kind='mergesort')[:, :neighbors]
    adjacency = np.zeros_like(weights)
    rows = np.arange(count)[:, None]
    adjacency[rows, selected] = weights[rows, selected]
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0)
    source, target = np.nonzero(adjacency)
    # Each matrix entry occurs once; symmetry creates exactly one reverse entry.
    edge_index = torch.from_numpy(np.stack([source, target]).astype(np.int64))
    edge_attr = torch.tensor(adjacency[source, target], dtype=torch.float32)
    pairs = set(zip(source.tolist(), target.tolist()))
    if any(i == j or (j, i) not in pairs for i, j in pairs):
        raise AssertionError('Similarity graph must be bidirectional without self loops')
    return Data(x=torch.tensor(similarity, dtype=torch.float32),
                edge_index=edge_index, edge_attr=edge_attr)


def clustered_hypergraph(similarity, clusters):
    """H: one cluster hyperedge per occupied cluster; W=I, G=Dv^-1/2 H De^-1 H.T Dv^-1/2."""
    if not 1 <= clusters <= len(similarity):
        raise ValueError('clusters must be between 1 and node_count')
    assignments = KMeans(n_clusters=clusters, random_state=0, n_init=10).fit_predict(similarity)
    # Drop empty clusters when duplicate rows yield fewer occupied clusters.
    _, labels = np.unique(assignments, return_inverse=True)
    incidence = np.zeros((len(labels), int(labels.max()) + 1), dtype=np.float64)
    incidence[np.arange(len(labels)), labels] = 1
    node_degree = incidence.sum(axis=1)
    edge_degree = incidence.sum(axis=0)
    normalized = incidence / np.sqrt(node_degree[:, None])
    operator = (normalized / edge_degree[None, :]) @ normalized.T
    if not np.isfinite(operator).all():
        raise AssertionError('Nonfinite hypergraph propagation operator')
    return (torch.tensor(operator, dtype=torch.float32),
            torch.tensor(incidence, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))


def build_graphs(features, neighbors, device, clusters=9):
    circ_similarity = features['IM'].detach().cpu().numpy()
    drug_similarity = features['ID'].detach().cpu().numpy()
    circ_operator, circ_incidence, circ_clusters = clustered_hypergraph(circ_similarity, clusters)
    drug_operator, drug_incidence, drug_clusters = clustered_hypergraph(drug_similarity, clusters)
    return dict(A=features['A'].to(device),
                circ_features=features['IM'].to(device), drug_features=features['ID'].to(device),
                circ_kmeans=circ_operator.to(device), drug_kmeans=drug_operator.to(device),
                circ_incidence=circ_incidence, drug_incidence=drug_incidence,
                circ_cluster_labels=circ_clusters, drug_cluster_labels=drug_clusters,
                circ_graph=similarity_graph(circ_similarity, neighbors).to(device),
                drug_graph=similarity_graph(drug_similarity, neighbors).to(device))


def forward(model, graphs):
    return model(graphs['A'], graphs['circ_graph'], graphs['drug_graph'],
                 graphs['circ_kmeans'], graphs['drug_kmeans'])


import math
import numpy as np
from radvl import hypergraph_neighbors as hypergraph_neighbors
from radvl import hypergraph_clustering as hypergraph_clustering

def constructHW_knn(X,K_neigs,is_probH):

    """incidence matrix"""
    H = hypergraph_neighbors.construct_H_with_KNN(X,K_neigs,is_probH)

    G = hypergraph_neighbors._generate_G_from_H(H)

    return G


def constructHW_kmean(X,clusters):

    """incidence matrix"""
    H = hypergraph_clustering.construct_H_with_Kmeans(X,clusters)

    G = hypergraph_clustering._generate_G_from_H(H)

    return G


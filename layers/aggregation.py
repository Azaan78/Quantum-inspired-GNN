"""
aggregation.py
--------------
Milestone 2: "modular propagation framework" + "scalable graph support".

Milestone 1 built the adjacency structure straight into Node/Graph objects
(a node's `.neighbours` list and `.weights` dict) and looped over them in
Python. That's fine for a 4-node toy graph, but doesn't scale to a real
dataset like Cora (2,708 nodes, ~5,400 edges) - looping in pure Python over
every node/neighbour pair every epoch would be very slow.

This module instead works with a plain NumPy/SciPy adjacency matrix, which
is the standard representation used by virtually every real GNN
implementation (GCN, GAT, etc.), and is what makes Milestone 3's real
datasets practical.
"""

import numpy as np
import scipy.sparse as sp


def edges_to_adjacency(num_nodes, edge_index, edge_weight=None, directed=False):
    """
    Build a dense NxN adjacency matrix from an edge list.

    edge_index: array-like of shape (2, E) or (E, 2) - pairs (source, target)
    edge_weight: optional array of shape (E,); defaults to all-ones (an
        unweighted graph), matching most real citation/molecule datasets.
    directed: if False (default), every edge is added in both directions,
        matching Milestone 1's `Graph.connect_nodes`, which always added a
        bidirectional edge.
    """
    edge_index = np.asarray(edge_index)
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    src, dst = edge_index[0], edge_index[1]

    if edge_weight is None:
        edge_weight = np.ones(src.shape[0], dtype=np.float64)

    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    A[src, dst] = edge_weight
    if not directed:
        A[dst, src] = edge_weight
    return A


def add_self_loops(A):
    """Add an identity matrix so every node aggregates its own features too
    (otherwise a node with no self-loop would completely lose its own
    feature vector the moment it aggregates neighbour messages)."""
    return A + np.eye(A.shape[0])


def normalize_adjacency(A, add_self_loop=True):
    """
    Symmetric degree normalisation:

        A_hat = D^(-1/2) (A + I) D^(-1/2)

    This is the same normalisation used in Kipf & Welling's original GCN
    (2017) - it keeps the scale of aggregated features stable regardless of
    how many neighbours a node has, which matters a lot once the graph has
    thousands of nodes with wildly different degrees (Cora's node degrees
    range from 1 to 168). Without this, high-degree "hub" nodes would
    dominate training purely because they sum more terms, not because
    they're more informative.
    """
    A = np.asarray(A, dtype=np.float64)
    if add_self_loop:
        A = add_self_loops(A)

    degree = A.sum(axis=1)
    # Isolated nodes would give 1/sqrt(0); guard against that.
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)

    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def dense_to_sparse(A):
    """Convert to SciPy CSR format for the (much larger) datasets, where a
    dense NxN matrix would waste memory on mostly-zero entries. Used by the
    dataset loaders in utils/graph_builder.py during preprocessing; the
    quantum-interference layer itself still operates densely since the
    *interference* matrix (built from phase, not from A) is dense anyway."""
    return sp.csr_matrix(A)

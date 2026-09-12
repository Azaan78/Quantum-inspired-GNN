"""
update.py
---------
Milestone 2: the "update" half of message passing (aggregate, then update),
plus the readout functions Milestone 3's graph classification needs.

Milestone 1 only had one activation (ReLU) and one "update" rule (overwrite
node.energy with the new value). Here that's generalised into a small,
swappable set of activation functions (each with its derivative, since the
trainer needs it for backprop) and graph-level readout functions.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activations (each returns (output, local_gradient_fn) is overkill for what
# we need - instead we expose forward + derivative-given-input pairs, which
# is simpler to unit test and matches how training/trainer.py uses them).
# ---------------------------------------------------------------------------

def relu(z):
    """Milestone 1's activation, unchanged: max(0, z), applied elementwise."""
    return np.maximum(0.0, z)


def relu_grad(z):
    """d(relu)/dz - 1 where z > 0, else 0."""
    return (z > 0).astype(np.float64)


def identity(z):
    """No activation - used for the final layer before a softmax/logits head,
    where we don't want to zero out negative logits."""
    return z


def identity_grad(z):
    return np.ones_like(z)


ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "identity": (identity, identity_grad),
    "linear": (identity, identity_grad),
}


def get_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(ACTIVATIONS)}.")
    return ACTIVATIONS[name]


# ---------------------------------------------------------------------------
# Graph-level readout (needed for Milestone 3's graph classification, e.g.
# MUTAG: many small graphs, each needs ONE label, not one label per node).
# ---------------------------------------------------------------------------

def mean_readout(node_embeddings):
    """Average all node embeddings into a single graph-level vector.
    This is the standard, simplest GNN readout (used by e.g. GIN's
    baseline)."""
    return node_embeddings.mean(axis=0)


def sum_readout(node_embeddings):
    """Sum instead of mean - keeps graph size information (a 50-node graph
    produces a 'bigger' embedding than a 10-node graph), which mean_readout
    normalises away."""
    return node_embeddings.sum(axis=0)


def max_readout(node_embeddings):
    """Elementwise max across nodes - highlights the single most 'active'
    node per feature dimension, useful when a graph's label is driven by
    one distinctive substructure rather than an overall average."""
    return node_embeddings.max(axis=0)


READOUTS = {
    "mean": mean_readout,
    "sum": sum_readout,
    "max": max_readout,
}


def get_readout(name):
    if name not in READOUTS:
        raise ValueError(f"Unknown readout '{name}'. Choose from {list(READOUTS)}.")
    return READOUTS[name]

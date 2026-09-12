"""
models/quantum_gnn.py
----------------------
Milestone 2's "modular propagation framework": stacks
layers.message_passing.QuantumMessagePassingLayer objects into an actual
multi-layer network, and Milestone 3's model heads for node classification
(Cora/CiteSeer/PubMed) and graph classification (MUTAG).

Why stacking layers matters for the "non-local learning rule":
Milestone 1 only had ONE round of message passing, so a node's update only
ever depended on its immediate (1-hop) neighbours - a purely local rule.
Stacking L layers here means information (and, during backprop, gradient)
travels L hops across the graph, so by the time you've trained a 2- or
3-layer QuantumGNN, a node's phase and weights are being updated based on
error signal from nodes several hops away, not just its direct neighbours.
That's the "non-local learning rule" the project's open items called for -
it falls out naturally from stacking the formal layer, rather than needing
a bespoke mechanism.
"""

import numpy as np

from layers.message_passing import QuantumMessagePassingLayer
from layers.update import get_readout


class QuantumGNN:
    """
    A stack of QuantumMessagePassingLayer objects sharing ONE phase vector
    (one phase per node in the graph, not one per layer - phase is a
    property of the node/qubit itself, matching Milestone 1's model, where
    `node.phase` was a single value used throughout `forward_propagate`).

    layer_dims: e.g. [in_features, hidden, out_features] for a 2-layer net.
    """

    def __init__(self, layer_dims, activation="relu", kernel="cosine",
                 decoherence_rate=0.0, seed=None):
        if len(layer_dims) < 2:
            raise ValueError("Need at least [in_features, out_features].")

        rng_seed_seq = np.random.default_rng(seed)
        self.layers = []
        for i in range(len(layer_dims) - 1):
            is_last = (i == len(layer_dims) - 2)
            layer_seed = int(rng_seed_seq.integers(0, 2**31 - 1))
            self.layers.append(
                QuantumMessagePassingLayer(
                    layer_dims[i], layer_dims[i + 1],
                    # Keep the nonlinearity between hidden layers, but leave
                    # the final layer's output as raw logits (identity) so
                    # training/loss.py's softmax+cross-entropy is well-defined.
                    activation=activation if not is_last else "identity",
                    kernel=kernel,
                    decoherence_rate=decoherence_rate,
                    seed=layer_seed,
                )
            )

    def forward(self, X, A_norm, phase):
        """X: (N, in_features) -> returns (N, out_features)."""
        H = X
        for depth, layer in enumerate(self.layers):
            H = layer.forward(H, A_norm, phase, depth=depth)
        return H

    def backward(self, dL_dHout):
        """
        Backprop through every layer. Returns:
          layer_grads: list of {"W":.., "b":..} per layer, in forward order
          phase_grad: (N,) - SUMMED across every layer, because all layers
                      share the same phase vector (this sum is exactly what
                      makes the update non-local - see module docstring)
          dL_dX: gradient with respect to the original input features
                 (unused for training but useful for feature-importance
                 style analysis, so it's returned rather than discarded)
        """
        layer_grads = []
        phase_grad = None
        dL_dH = dL_dHout
        for layer in reversed(self.layers):
            grads, dL_dH = layer.backward(dL_dH)
            layer_grads.append({"W": grads["W"], "b": grads["b"]})
            phase_grad = grads["phase"] if phase_grad is None else phase_grad + grads["phase"]
        layer_grads.reverse()
        return layer_grads, phase_grad, dL_dH

    def parameters(self):
        return [layer.parameters() for layer in self.layers]

    def apply_updates(self, layer_deltas):
        for layer, delta in zip(self.layers, layer_deltas):
            layer.apply_updates(delta["W"], delta["b"])


class LinearHead:
    """
    A plain linear classifier head: embedding_dim -> num_classes.

    Used on top of QuantumGNN's node embeddings for node classification
    (Cora/CiteSeer/PubMed), and on top of a pooled graph embedding for
    graph classification (MUTAG). Kept separate from QuantumGNN itself so
    the same GNN backbone can be reused for either task - exactly the
    "modular" framework Milestone 2 asked for.
    """

    def __init__(self, in_features, num_classes, seed=None):
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (in_features + num_classes))
        self.W = rng.uniform(-limit, limit, size=(in_features, num_classes))
        self.b = np.zeros(num_classes, dtype=np.float64)
        self._cache = None

    def forward(self, H):
        self._cache = H
        return H @ self.W + self.b

    def backward(self, dL_dlogits):
        H = self._cache
        dL_dW = H.T @ dL_dlogits
        dL_db = dL_dlogits.sum(axis=0)
        dL_dH = dL_dlogits @ self.W.T
        return {"W": dL_dW, "b": dL_db}, dL_dH

    def parameters(self):
        return {"W": self.W, "b": self.b}

    def apply_updates(self, delta_W, delta_b):
        self.W = self.W + delta_W
        self.b = self.b + delta_b


class GraphClassifier:
    """
    Milestone 3's graph-classification model (e.g. MUTAG): run the QuantumGNN
    backbone over EACH graph individually, pool its node embeddings into one
    graph-level vector with a readout function, then classify that vector
    with a LinearHead. One label per graph, not per node.
    """

    def __init__(self, layer_dims, num_classes, activation="relu",
                 kernel="cosine", decoherence_rate=0.0, readout="mean", seed=None):
        self.gnn = QuantumGNN(layer_dims, activation=activation, kernel=kernel,
                               decoherence_rate=decoherence_rate, seed=seed)
        self.head = LinearHead(layer_dims[-1], num_classes, seed=seed)
        self.readout_name = readout
        self._readout_fn = get_readout(readout)
        self._last_n_nodes = None

    def forward(self, X, A_norm, phase):
        H = self.gnn.forward(X, A_norm, phase)       # (N, embed_dim)
        self._last_n_nodes = H.shape[0]
        graph_embedding = self._readout_fn(H)          # (embed_dim,)
        logits = self.head.forward(graph_embedding[None, :])[0]  # (num_classes,)
        return logits

    def backward(self, dL_dlogits):
        head_grads, dL_dgraph_embedding = self.head.backward(dL_dlogits[None, :])
        n = self._last_n_nodes
        # Undo the readout: distribute the graph-level gradient back across
        # every node's embedding. For mean/sum readout this has a simple
        # closed form; for max readout we'd need to route it only to the
        # argmax node per dimension - mean/sum cover the readouts actually
        # used in training/trainer.py's GraphClassifier training loop.
        if self.readout_name == "mean":
            dL_dH = np.tile(dL_dgraph_embedding[0] / n, (n, 1))
        elif self.readout_name == "sum":
            dL_dH = np.tile(dL_dgraph_embedding[0], (n, 1))
        else:
            raise NotImplementedError(
                "Backprop through 'max' readout isn't implemented; use "
                "'mean' or 'sum' for training."
            )

        layer_grads, phase_grad, _ = self.gnn.backward(dL_dH)
        return layer_grads, phase_grad, head_grads

    def parameters(self):
        return {"gnn": self.gnn.parameters(), "head": self.head.parameters()}

"""
message_passing.py
-------------------
Milestone 2's headline deliverable: "a formal message passing layer".

Milestone 1's forward_propagate() function did three things all mixed
together in one Python loop: aggregate a neighbour's energy weighted by an
edge weight AND a phase-interference term, sum it up, add a bias, then ReLU
it. This module pulls that apart into a proper, reusable *layer* - the same
shape as a layer in any modern GNN library - so it can be:

  * stacked (multiple layers = information travels multiple hops, which is
    what makes the "non-local learning rule" below possible)
  * trained with real backpropagation (Milestone 1 only had a hand-written,
    one-hop local update rule)
  * reused unchanged on any graph, including the real datasets in
    Milestone 3, not just the hand-built 4-qubit example

The layer's forward pass, in one line, is:

    H_out = activation( (A_hat * Interference(phase) * decoherence) @ H_in @ W + b )

where A_hat is the normalised adjacency matrix (layers/aggregation.py),
Interference(phase) is the quantum-inspired interference matrix
(layers/quantum_interference.py), and W/b are ordinary learnable weights -
exactly like a Graph Convolutional Network layer, except the adjacency
matrix itself is modulated every forward pass by the (learnable!) phase of
every node.
"""

import numpy as np

from layers.quantum_interference import interference_matrix, interference_gradient, decoherence_factor
from layers.update import get_activation


class QuantumMessagePassingLayer:
    """
    One quantum-inspired message-passing layer.

    Parameters
    ----------
    in_features, out_features : int
        Size of the node feature vectors coming in / going out. This is the
        direct answer to Milestone 2's "node feature vectors" objective:
        Milestone 1 only ever had a single scalar (`node.energy`) per node;
        here every node carries a full feature vector, and this layer's job
        is to transform (in_features,) -> (out_features,) per node while
        mixing in information from its neighbours.
    activation : str
        One of layers.update.ACTIVATIONS ("relu", "identity").
    kernel : str
        Which interference function to use (layers/quantum_interference.py):
        "cosine" (Milestone 2, default, and the only one with a trained
        gradient), "squared_cosine" or "gaussian" (Milestone 4 alternatives,
        forward-pass only - see quantum_interference.interference_gradient).
    decoherence_rate : float
        Milestone 4's decoherence-inspired damping. 0.0 (default) disables
        it, reproducing plain Milestone 2/3 behaviour.
    """

    def __init__(self, in_features, out_features, activation="relu",
                 kernel="cosine", decoherence_rate=0.0, seed=None):
        rng = np.random.default_rng(seed)

        # Glorot/Xavier-style initialisation: keeps the variance of
        # activations roughly stable as they pass through the layer,
        # regardless of in_features/out_features - standard practice, and
        # important here because Cora's raw features are 1,433-dimensional.
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = rng.uniform(-limit, limit, size=(in_features, out_features))
        self.b = np.zeros(out_features, dtype=np.float64)

        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        self.kernel = kernel
        self.decoherence_rate = decoherence_rate
        self._act_fn, self._act_grad_fn = get_activation(activation)

        self._cache = None

    def forward(self, H, A_norm, phase, depth=0):
        """
        H : (N, in_features) node feature matrix
        A_norm : (N, N) normalised adjacency (layers/aggregation.py)
        phase : (N,) per-node phase (shared across all layers in the model -
            see models/quantum_gnn.py)
        depth : which layer index this is in the stack, used only for the
            decoherence damping term.

        Returns H_out : (N, out_features)
        """
        I = interference_matrix(phase, kernel=self.kernel)
        damping = decoherence_factor(depth, self.decoherence_rate)

        # The quantum-modulated propagation matrix: normal graph structure,
        # reweighted node-pair-by-node-pair by how "in phase" the two nodes
        # currently are, further damped by decoherence if enabled.
        A_q = A_norm * I * damping

        M = A_q @ H                      # (N, in_features) - aggregated messages
        Z = M @ self.W + self.b          # (N, out_features) - linear transform
        H_out = self._act_fn(Z)          # (N, out_features) - nonlinearity

        self._cache = dict(H=H, A_norm=A_norm, I=I, damping=damping,
                            A_q=A_q, M=M, Z=Z, phase=phase)
        return H_out

    def backward(self, dL_dHout):
        """
        Standard backprop through this layer, PLUS the extra term that makes
        this different from an ordinary GCN layer: gradient with respect to
        the phase vector that produced the interference matrix.

        Returns (grads, dL_dH) where:
          grads = {"W": ..., "b": ..., "phase": ...}
          dL_dH = gradient to keep propagating into the previous layer / the
                  original input features.
        """
        c = self._cache
        if c is None:
            raise RuntimeError("Call forward() before backward().")

        dL_dZ = dL_dHout * self._act_grad_fn(c["Z"])         # (N, out_features)
        dL_dW = c["M"].T @ dL_dZ                              # (in_features, out_features)
        dL_db = dL_dZ.sum(axis=0)                             # (out_features,)
        dL_dM = dL_dZ @ self.W.T                              # (N, in_features)

        # M = A_q @ H
        dL_dAq = dL_dM @ c["H"].T                              # (N, N)
        dL_dH = c["A_q"].T @ dL_dM                             # (N, in_features)

        # A_q = A_norm * I * damping  (elementwise) -> dL/dI = dL/dA_q * A_norm * damping
        dL_dI = dL_dAq * c["A_norm"] * c["damping"]

        if self.kernel == "cosine":
            dL_dphase = interference_gradient(c["phase"], dL_dI, kernel="cosine")
        else:
            # Milestone 4's alternative kernels are supported for forward-pass
            # experimentation; phase is simply not updated through them yet.
            dL_dphase = np.zeros_like(c["phase"])

        grads = {"W": dL_dW, "b": dL_db, "phase": dL_dphase}
        return grads, dL_dH

    def parameters(self):
        return {"W": self.W, "b": self.b}

    def apply_updates(self, delta_W, delta_b):
        self.W = self.W + delta_W
        self.b = self.b + delta_b

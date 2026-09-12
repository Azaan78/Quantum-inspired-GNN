"""
tests/test_layers.py
---------------------
Milestone 2's unit tests. These check three things a research-grade
component needs and Milestone 1's script had no way to check:

  1. Shapes and basic properties are correct (interference bounded in
     [-1, 1], decoherence never amplifies, etc.)
  2. The hand-derived backward() gradients actually match the true
     gradient, verified numerically (this is the standard way to catch
     backprop bugs - compare the analytic gradient against a finite
     difference approximation).
  3. The whole pipeline runs end-to-end and genuinely learns something
     better than chance on structured data.

Run with:  python -m pytest tests/ -v
"""

import numpy as np
import pytest

from layers.quantum_interference import (
    cosine_interference, squared_cosine_interference, gaussian_phase_interference,
    interference_gradient, decoherence_factor, complex_interference_matrix,
)
from layers.aggregation import edges_to_adjacency, normalize_adjacency
from layers.message_passing import QuantumMessagePassingLayer
from models.quantum_gnn import QuantumGNN, GraphClassifier
from training.loss import softmax_cross_entropy, accuracy, mse_loss
from training.trainer import NodeClassifierTrainer, GraphClassifierTrainer
from utils.graph_builder import (
    synthetic_citation_graph, synthetic_graph_classification_dataset,
)


# ---------------------------------------------------------------------------
# 1. Basic properties
# ---------------------------------------------------------------------------

def test_cosine_interference_bounds_and_self_term():
    phase = np.array([0.0, 1.2, 2.1, 0.8, -3.0])
    I = cosine_interference(phase)
    assert I.shape == (5, 5)
    assert np.allclose(np.diag(I), 1.0), "a node must be perfectly in phase with itself"
    assert np.all(I <= 1.0 + 1e-9) and np.all(I >= -1.0 - 1e-9)
    assert np.allclose(I, I.T), "cosine interference must be symmetric"


def test_squared_cosine_interference_is_nonnegative():
    phase = np.random.default_rng(0).uniform(-5, 5, size=20)
    I = squared_cosine_interference(phase)
    assert np.all(I >= -1e-9), "squared-cosine interference should never go negative"
    assert np.all(I <= 1.0 + 1e-9)


def test_gaussian_interference_peaks_at_zero_difference():
    phase = np.array([0.0, 0.0, 5.0])
    I = gaussian_phase_interference(phase, sigma=1.0)
    assert np.isclose(I[0, 1], 1.0), "identical phase should give maximal interference"
    assert I[0, 2] < I[0, 1], "larger phase gap should give weaker interference"


def test_decoherence_factor_decays_with_depth():
    f0 = decoherence_factor(depth=0, decoherence_rate=0.3)
    f1 = decoherence_factor(depth=1, decoherence_rate=0.3)
    f2 = decoherence_factor(depth=2, decoherence_rate=0.3)
    assert f0 == 1.0, "no decoherence at depth 0"
    assert f0 > f1 > f2 > 0, "decoherence should strictly decay with depth"


def test_zero_decoherence_rate_is_a_no_op():
    for depth in range(5):
        assert decoherence_factor(depth, decoherence_rate=0.0) == 1.0


def test_complex_interference_reduces_to_cosine_when_magnitude_is_one():
    phase = np.random.default_rng(1).uniform(-3, 3, size=8)
    magnitude = np.ones_like(phase)
    complex_I = complex_interference_matrix(magnitude, phase)
    cos_I = cosine_interference(phase)
    assert np.allclose(complex_I, cos_I, atol=1e-10)


def test_adjacency_normalisation_is_symmetric_and_self_connected():
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    A = edges_to_adjacency(4, edges)
    A_norm = normalize_adjacency(A)
    assert np.allclose(A_norm, A_norm.T)
    assert np.all(np.diag(A_norm) > 0), "self-loops mean every node keeps a bit of itself"


# ---------------------------------------------------------------------------
# 2. Gradient checking - the single most important correctness test here.
# ---------------------------------------------------------------------------

def _numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f()
        x[idx] = orig - eps
        f_minus = f()
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def test_message_passing_layer_weight_gradient_matches_numerical():
    rng = np.random.default_rng(42)
    n, f_in, f_out = 6, 4, 3
    X = rng.normal(size=(n, f_in))
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5], [1, 4]])
    A = edges_to_adjacency(n, edges)
    A_norm = normalize_adjacency(A)
    phase = rng.uniform(-np.pi, np.pi, size=n)

    layer = QuantumMessagePassingLayer(f_in, f_out, activation="relu", seed=0)

    def loss_fn():
        out = layer.forward(X, A_norm, phase)
        return float(np.sum(out ** 2))  # arbitrary scalar loss

    out = layer.forward(X, A_norm, phase)
    dL_dHout = 2 * out  # gradient of sum(out**2) w.r.t. out
    grads, _ = layer.backward(dL_dHout)

    numerical_W = _numerical_gradient(loss_fn, layer.W)
    assert np.allclose(grads["W"], numerical_W, atol=1e-4), "analytic dL/dW must match finite differences"


def test_message_passing_layer_phase_gradient_matches_numerical():
    rng = np.random.default_rng(7)
    n, f_in, f_out = 5, 3, 2
    X = rng.normal(size=(n, f_in))
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]])
    A = edges_to_adjacency(n, edges)
    A_norm = normalize_adjacency(A)
    phase = rng.uniform(-np.pi, np.pi, size=n)

    layer = QuantumMessagePassingLayer(f_in, f_out, activation="identity", seed=1)

    def loss_fn():
        out = layer.forward(X, A_norm, phase)
        return float(np.sum(out ** 2))

    out = layer.forward(X, A_norm, phase)
    dL_dHout = 2 * out
    grads, _ = layer.backward(dL_dHout)

    numerical_phase = _numerical_gradient(loss_fn, phase)
    assert np.allclose(grads["phase"], numerical_phase, atol=1e-4), (
        "analytic dL/dphase must match finite differences - this is the gradient "
        "that makes phase learnable / gives the non-local update rule"
    )


def test_stacked_gnn_gradient_matches_numerical():
    """A 2-layer stack, checking the *shared* phase gradient (summed across
    both layers) is still correct - this is the specific claim the project
    notes make about a 'non-local learning rule'."""
    rng = np.random.default_rng(3)
    n = 6
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5]])
    A = edges_to_adjacency(n, edges)
    A_norm = normalize_adjacency(A)
    X = rng.normal(size=(n, 4))
    phase = rng.uniform(-np.pi, np.pi, size=n)

    model = QuantumGNN([4, 3, 2], activation="relu", seed=5)

    def loss_fn():
        out = model.forward(X, A_norm, phase)
        return float(np.sum(out ** 2))

    out = model.forward(X, A_norm, phase)
    _, phase_grad, _ = model.backward(2 * out)
    numerical_phase = _numerical_gradient(loss_fn, phase)
    assert np.allclose(phase_grad, numerical_phase, atol=1e-4)


# ---------------------------------------------------------------------------
# 3. End-to-end: does it actually learn?
# ---------------------------------------------------------------------------

def test_node_classifier_beats_random_chance_on_synthetic_citation_graph():
    data = synthetic_citation_graph(num_nodes=200, num_features=48, num_classes=4, seed=0)
    model = QuantumGNN([48, 16, 4], activation="relu", seed=0)
    phase_init = np.random.default_rng(0).uniform(-np.pi, np.pi, size=200)
    trainer = NodeClassifierTrainer(model, phase_init, lr=0.05, optimizer="adam")

    result = trainer.fit(
        data["X"], data["A_norm"], data["labels"],
        data["train_mask"], data["val_mask"], data["test_mask"],
        epochs=60, verbose_every=0,
    )
    random_chance = 1.0 / 4
    assert result["test_acc"] > random_chance + 0.15, (
        f"expected meaningfully better than chance ({random_chance:.2f}), got {result['test_acc']:.2f}"
    )
    assert result["history"]["loss"][-1] < result["history"]["loss"][0], "loss should decrease overall"


def test_graph_classifier_beats_random_chance_on_synthetic_mutag_like_data():
    dataset = synthetic_graph_classification_dataset(num_graphs=80, seed=1)
    train, test = dataset[:60], dataset[60:]

    model = GraphClassifier([7, 8, 8], num_classes=2, activation="relu", readout="mean", seed=0)
    trainer = GraphClassifierTrainer(model, lr=0.05, optimizer="adam")
    trainer.fit(train, epochs=40, verbose_every=0)
    test_acc = trainer.evaluate(test)

    assert test_acc > 0.5, f"expected better than coin-flip on a binary task, got {test_acc:.2f}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

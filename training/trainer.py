"""
training/trainer.py
--------------------
Milestone 1's "training loop" was Single_step_training() in main.py: four
hard-coded iterations, printing every intermediate value for manual
debugging. This module replaces that with two reusable trainers -
NodeClassifierTrainer (Cora/CiteSeer/PubMed) and GraphClassifierTrainer
(MUTAG) - that report real metrics (loss + accuracy on proper train/val/test
splits) instead of print-statement debugging.
"""

import numpy as np

from training.loss import softmax_cross_entropy, accuracy
from training.optimizer import Adam, SGD


def _make_optimizer(name, lr):
    if name == "adam":
        return Adam(lr=lr)
    if name == "sgd":
        return SGD(lr=lr)
    raise ValueError(f"Unknown optimizer '{name}'. Choose 'adam' or 'sgd'.")


class NodeClassifierTrainer:
    """
    Trains a models.quantum_gnn.QuantumGNN whose final layer outputs
    (N, num_classes) logits directly, on a SINGLE graph (Cora/CiteSeer/
    PubMed are each one big citation graph, not many small graphs).

    The phase vector is a genuine trainable parameter here (see
    layers/quantum_interference.py's interference_gradient) - this is what
    replaces Milestone 1's random phase *drift* with an actual learned
    phase, and is shared across every layer in the stack, which is what
    makes its gradient a "non-local" signal (module docstring in
    models/quantum_gnn.py has the full explanation).
    """

    def __init__(self, model, phase_init, lr=0.01, optimizer="adam"):
        self.model = model
        self.phase = np.array(phase_init, dtype=np.float64, copy=True)
        self.optimizer_name = optimizer
        self.opt = _make_optimizer(optimizer, lr)

    def fit(self, X, A_norm, labels, train_mask, val_mask=None, test_mask=None,
            epochs=100, verbose_every=10):
        history = {"loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            logits = self.model.forward(X, A_norm, self.phase)
            loss, dL_dlogits = softmax_cross_entropy(logits, labels, mask=train_mask)
            layer_grads, phase_grad, _ = self.model.backward(dL_dlogits)

            for i, layer in enumerate(self.model.layers):
                dW = self.opt.step(f"layer{i}_W", layer_grads[i]["W"])
                db = self.opt.step(f"layer{i}_b", layer_grads[i]["b"])
                layer.apply_updates(dW, db)

            dphase = self.opt.step("phase", phase_grad)
            self.phase = self.phase + dphase

            train_acc = accuracy(logits, labels, train_mask)
            val_acc = accuracy(logits, labels, val_mask) if val_mask is not None else float("nan")

            history["loss"].append(loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if verbose_every and (epoch % verbose_every == 0 or epoch == 1):
                print(f"epoch {epoch:4d} | loss {loss:.4f} | train acc {train_acc:.3f} | val acc {val_acc:.3f}")

        result = {"history": history}
        if test_mask is not None:
            final_logits = self.model.forward(X, A_norm, self.phase)
            result["test_acc"] = accuracy(final_logits, labels, test_mask)
        return result


class GraphClassifierTrainer:
    """
    Trains a models.quantum_gnn.GraphClassifier over a DATASET of many small
    graphs (MUTAG: 188 graphs, each with its own nodes/edges/label).

    Design note (documented honestly rather than hidden): each graph gets
    its own phase vector, initialised deterministically from that graph's
    node degrees (so it's reproducible and structurally meaningful rather
    than arbitrary) but NOT trained in this milestone. Training a
    per-graph phase alongside weights shared across a whole dataset raises
    questions (does phase generalise across different molecules, or is it
    graph-specific structure?) that are flagged as Milestone 5+ follow-up
    in the README rather than answered with an under-tested guess here.
    """

    def __init__(self, model, lr=0.01, optimizer="adam"):
        self.model = model
        self.optimizer_name = optimizer
        self.opt = _make_optimizer(optimizer, lr)

    @staticmethod
    def degree_phase(A_norm):
        """Deterministic per-graph phase init: scale each node's normalised
        degree into [0, 2*pi). Two structurally identical graphs get the
        same phase pattern, which is what "deterministic" buys us here."""
        degree = A_norm.sum(axis=1)
        if degree.max() > 0:
            degree = degree / degree.max()
        return degree * 2 * np.pi

    def fit(self, dataset, epochs=30, verbose_every=5):
        """
        dataset: list of dicts, each {"X":.., "A_norm":.., "label": int}
        (see utils/graph_builder.py for how MUTAG is turned into this shape).
        """
        history = {"loss": [], "acc": []}

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            correct = 0

            for graph in dataset:
                phase = self.degree_phase(graph["A_norm"])
                logits = self.model.forward(graph["X"], graph["A_norm"], phase)
                loss, dL_dlogits = softmax_cross_entropy(
                    logits[None, :], np.array([graph["label"]])
                )
                epoch_loss += loss
                correct += int(logits.argmax() == graph["label"])

                layer_grads, _phase_grad, head_grads = self.model.backward(dL_dlogits[0])

                for i, layer in enumerate(self.model.gnn.layers):
                    dW = self.opt.step(f"layer{i}_W", layer_grads[i]["W"])
                    db = self.opt.step(f"layer{i}_b", layer_grads[i]["b"])
                    layer.apply_updates(dW, db)

                dW_head = self.opt.step("head_W", head_grads["W"])
                db_head = self.opt.step("head_b", head_grads["b"])
                self.model.head.apply_updates(dW_head, db_head)

            mean_loss = epoch_loss / len(dataset)
            acc = correct / len(dataset)
            history["loss"].append(mean_loss)
            history["acc"].append(acc)

            if verbose_every and (epoch % verbose_every == 0 or epoch == 1):
                print(f"epoch {epoch:4d} | mean loss {mean_loss:.4f} | train acc {acc:.3f}")

        return {"history": history}

    def evaluate(self, dataset):
        correct = 0
        for graph in dataset:
            phase = self.degree_phase(graph["A_norm"])
            logits = self.model.forward(graph["X"], graph["A_norm"], phase)
            correct += int(logits.argmax() == graph["label"])
        return correct / len(dataset) if dataset else 0.0

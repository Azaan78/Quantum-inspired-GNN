"""
demo_milestones_2_3_4.py
--------------------------
A single script that exercises everything built for Milestones 2, 3 and
(partially) 4, and produces the plots/numbers written up in
docs/RESULTS.md. Kept deliberately separate from main.py, which remains
the original Milestone 1 demonstration, unedited.

Run with:  python demo_milestones_2_3_4.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from models.quantum_gnn import QuantumGNN, GraphClassifier
from training.trainer import NodeClassifierTrainer, GraphClassifierTrainer
from utils.graph_builder import (
    synthetic_citation_graph, synthetic_graph_classification_dataset,
)
from layers.quantum_interference import (
    cosine_interference, squared_cosine_interference, gaussian_phase_interference,
)

RESULTS = {}


def milestone_2_and_3_node_classification():
    print("\n=== Milestone 2 + 3: node classification (synthetic, Cora-shaped) ===")
    data = synthetic_citation_graph(num_nodes=500, num_features=100, num_classes=6, seed=0)
    model = QuantumGNN([100, 32, 6], activation="relu", seed=0)
    phase_init = np.random.default_rng(0).uniform(-np.pi, np.pi, size=500)
    trainer = NodeClassifierTrainer(model, phase_init, lr=0.05, optimizer="adam")

    result = trainer.fit(
        data["X"], data["A_norm"], data["labels"],
        data["train_mask"], data["val_mask"], data["test_mask"],
        epochs=150, verbose_every=25,
    )

    RESULTS["node_classification"] = {
        "num_nodes": 500, "num_classes": 6,
        "final_train_acc": result["history"]["train_acc"][-1],
        "final_val_acc": result["history"]["val_acc"][-1],
        "test_acc": result["test_acc"],
        "random_chance": 1 / 6,
    }

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(result["history"]["loss"])
    ax_loss.set_title("Node classification - training loss")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("cross-entropy loss"); ax_loss.grid(True)

    ax_acc.plot(result["history"]["train_acc"], label="train")
    ax_acc.plot(result["history"]["val_acc"], label="val")
    ax_acc.axhline(1 / 6, color="grey", linestyle="--", label="random chance")
    ax_acc.set_title("Node classification - accuracy")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy"); ax_acc.legend(); ax_acc.grid(True)

    plt.tight_layout()
    plt.savefig("docs/milestone2_node_classification.png", dpi=130)
    plt.close()
    print(f"Test accuracy: {result['test_acc']:.3f} (random chance: {1/6:.3f})")


def milestone_3_graph_classification():
    print("\n=== Milestone 3: graph classification (synthetic, MUTAG-shaped) ===")
    dataset = synthetic_graph_classification_dataset(num_graphs=200, seed=1)
    train, test = dataset[:150], dataset[150:]

    model = GraphClassifier([7, 8, 8], num_classes=2, activation="relu", readout="mean", seed=0)
    trainer = GraphClassifierTrainer(model, lr=0.05, optimizer="adam")
    result = trainer.fit(train, epochs=40, verbose_every=10)
    test_acc = trainer.evaluate(test)

    RESULTS["graph_classification"] = {
        "num_graphs": 200, "train_size": 150, "test_size": 50,
        "final_train_acc": result["history"]["acc"][-1],
        "test_acc": test_acc,
        "random_chance": 0.5,
    }

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(result["history"]["loss"])
    ax_loss.set_title("Graph classification - training loss")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("cross-entropy loss"); ax_loss.grid(True)

    ax_acc.plot(result["history"]["acc"], label="train")
    ax_acc.axhline(0.5, color="grey", linestyle="--", label="random chance")
    ax_acc.axhline(test_acc, color="green", linestyle=":", label=f"final test ({test_acc:.2f})")
    ax_acc.set_title("Graph classification - accuracy")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy"); ax_acc.legend(); ax_acc.grid(True)

    plt.tight_layout()
    plt.savefig("docs/milestone3_graph_classification.png", dpi=130)
    plt.close()
    print(f"Test accuracy: {test_acc:.3f} (random chance: 0.500)")


def milestone_4_interference_kernels():
    print("\n=== Milestone 4: alternative interference kernels ===")
    phase_diff = np.linspace(-2 * np.pi, 2 * np.pi, 400)
    # build a 2-node "phase" pair sweeping the difference, reuse the kernels
    # directly on a synthetic phase-difference axis for a clean plot.
    cos_vals = np.cos(phase_diff)
    sq_cos_vals = np.cos(phase_diff) ** 2
    gauss_vals = np.exp(-(phase_diff ** 2) / (2 * 1.0 ** 2))

    plt.figure(figsize=(7, 4))
    plt.plot(phase_diff, cos_vals, label="cosine (Milestone 2)")
    plt.plot(phase_diff, sq_cos_vals, label="squared cosine (Milestone 4)")
    plt.plot(phase_diff, gauss_vals, label="gaussian, sigma=1 (Milestone 4)")
    plt.xlabel("phase difference (radians)")
    plt.ylabel("interference weight")
    plt.title("Interference kernels compared")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("docs/milestone4_interference_kernels.png", dpi=130)
    plt.close()
    RESULTS["interference_kernels"] = "see docs/milestone4_interference_kernels.png"


def milestone_4_decoherence_effect():
    print("\n=== Milestone 4: decoherence-inspired damping ===")
    data = synthetic_citation_graph(num_nodes=300, num_features=80, num_classes=5, seed=2)
    rates = [0.0, 0.15, 0.5]
    scores = {}
    for rate in rates:
        model = QuantumGNN([80, 24, 5], activation="relu", decoherence_rate=rate, seed=1)
        phase_init = np.random.default_rng(1).uniform(-np.pi, np.pi, size=300)
        trainer = NodeClassifierTrainer(model, phase_init, lr=0.05, optimizer="adam")
        result = trainer.fit(
            data["X"], data["A_norm"], data["labels"],
            data["train_mask"], data["val_mask"], data["test_mask"],
            epochs=100, verbose_every=0,
        )
        scores[rate] = result["test_acc"]
        print(f"decoherence_rate={rate:<5} -> test accuracy {result['test_acc']:.3f}")

    RESULTS["decoherence_effect"] = scores


if __name__ == "__main__":
    milestone_2_and_3_node_classification()
    milestone_3_graph_classification()
    milestone_4_interference_kernels()
    milestone_4_decoherence_effect()

    with open("docs/milestone_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\nAll results written to docs/milestone_results.json")

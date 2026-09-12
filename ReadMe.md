# Quantum-Inspired Graph Learning Framework

> **Research Prototype | Quantum-Inspired Machine Learning | Graph Neural Networks | Dynamical Systems**

An experimental framework investigating whether quantum-inspired phase dynamics can enhance graph-based learning systems through phase-aware message passing, adaptive graph learning, and dynamic state evolution.

---

> ⚠️ **Current Status**
>
> This repository has completed **Milestone 2 (Formal GNN Layer)** and **Milestone 3 (Real Dataset Integration)** in full, and part of **Milestone 4 (Advanced Quantum-Inspired Dynamics)**.
>
> The original Milestone 1 prototype (`Node.py`, `Graph.py`, `main.py`, `graph_representation.py`) is untouched and still runs standalone. All new work lives in `layers/`, `models/`, `training/`, `utils/`, and `tests/`, and is documented in full in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** (the design/maths) and **[docs/RESULTS.md](docs/RESULTS.md)** (what was actually run, and the honest caveats around it).

---

## Contents

* [Overview](#overview)
* [Motivation](#motivation)
* [Why This Project?](#why-this-project)
* [Milestone 1 Features (original prototype)](#current-features)
* [Milestones 2-4: What Was Added](#milestones-2-4-what-was-added)
* [Architecture](#architecture)
* [Visualisation Dashboard](#visualisation-dashboard)
* [Research Questions](#research-questions)
* [Progress & Milestones](#progress--milestones)
* [Roadmap](#roadmap)
* [Installation](#installation)
* [Repository Structure](#repository-structure)
* [Future Research Directions](#future-research-directions)
* [Author](#author)
* [License](#license)

---

# Overview

The **Quantum-Inspired Graph Learning Framework** is an exploratory machine learning project investigating the intersection of:

* Quantum Computing
* Graph Neural Networks (GNNs)
* Dynamical Systems
* Artificial Intelligence

The framework models a graph of interconnected computational nodes where each node maintains internal quantum-inspired state variables:

* Energy / Feature vector (generalised from a single scalar to a full vector in Milestone 2)
* Phase (now a *trained* parameter as of Milestone 2 - see docs/ARCHITECTURE.md)
* Trainable Weights
* Trainable Biases

Information propagates through the graph using weighted neighbour interactions, while phase-dependent interference dynamically modulates communication between nodes.

The long-term goal is to evolve this prototype into a full **Quantum-Inspired Graph Neural Network (QGNN)** capable of learning from real-world graph datasets and benchmarking against traditional GNN architectures. Milestones 2 and 3 are the first concrete steps toward that goal: a formal, stackable layer, and real classification tasks (node classification and graph classification) rather than the single hand-built toy example.

---

# Motivation

As a Computer Science student with interests in both Artificial Intelligence and Quantum Computing, I wanted to challenge myself by building a project that combines concepts from both fields.

This project serves as both:

* A learning platform for understanding graph-based machine learning and quantum-inspired computation.
* A long-term research project exploring novel approaches to graph learning.

The framework is intended to continuously evolve as new concepts, techniques, and experimental results are incorporated.

---

# Why This Project?

Traditional Graph Neural Networks primarily rely on neighbourhood aggregation and learned feature transformations.

This project explores whether introducing quantum-inspired concepts such as:

* Phase
* Interference
* Dynamic state evolution
* Stochastic perturbation

can influence learning behaviour and potentially provide richer graph representations.

Rather than attempting to simulate a true quantum computer, this framework investigates whether useful learning mechanisms can emerge from **quantum-inspired dynamics implemented on classical hardware**.

---

# Current Features

*(Milestone 1 - the original prototype, `Node.py` / `Graph.py` / `main.py` / `graph_representation.py`, unchanged)*

## Graph-Based Architecture

* Custom graph implementation
* Custom node implementation
* Weighted graph connections
* Bidirectional neighbour communication

## Quantum-Inspired Node Dynamics

Each node maintains:

* Energy State
* Phase State
* Trainable Bias
* Trainable Edge Weights

## Phase-Dependent Interference

Neighbour interactions are influenced by:

```text
Interference = cos(phase_difference)
```

allowing message propagation strength to vary according to phase relationships between connected nodes.

## Learning System

Current implementation includes:

* Forward propagation
* Local error-driven learning updates
* Mean Squared Error (MSE) loss
* Adaptive weight updates
* Adaptive bias updates

## Dynamic Phase Evolution

The framework currently incorporates stochastic phase drift to simulate evolving quantum-inspired behaviour.

## Visualisation Dashboard

Automatically generates:

* Initial Graph State
* Final Graph State
* Training Loss Curve
* Phase Evolution Plot

---

# Milestones 2-4: What Was Added

*(New code, `layers/` / `models/` / `training/` / `utils/` / `tests/`. Full technical writeup: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).)*

* **A formal, stackable message-passing layer** (`layers/message_passing.py`) generalising Milestone 1's single hand-written propagation step into a reusable layer, usable for graphs of any size - not just the original 4-qubit example.
* **Node feature vectors** - every node now carries a full feature vector, not a single scalar `energy`.
* **Phase as a learnable parameter** - phase is no longer pure random drift; it has an analytic gradient (`layers/quantum_interference.interference_gradient`, verified numerically in `tests/`) and is updated by backpropagation like any other parameter.
* **A non-local learning rule** - because phase is shared across every layer in a stack, its gradient sums contributions from errors several hops away, not just a node's immediate neighbour (full derivation in `docs/ARCHITECTURE.md`, section 3).
* **Real dataset integration** (Milestone 3): loaders for Cora, CiteSeer, PubMed (node classification) and MUTAG (graph classification) in `utils/graph_builder.py`, plus a full training/evaluation pipeline for both task types (`training/trainer.py`). See `docs/RESULTS.md` for an important caveat on what was actually run against real vs. synthetic data.
* **Milestone 4 (partial)**: alternative interference kernels, a decoherence-inspired damping term, and a complex-valued amplitude representation - see the table in `docs/ARCHITECTURE.md`, section 5, for exactly what's done vs. left as future work.
* **Unit tests with gradient checking** (`tests/test_layers.py`) - every custom backward pass is checked against a numerical (finite-difference) gradient, not just "loss goes down".

---

# Architecture

Milestone 1's propagation pipeline (unchanged, still runs via `main.py`):

```text
Input Graph
     │
     ▼
Neighbour Aggregation
     │
     ▼
Phase Interference Modulation
     │
     ▼
Weighted Message Passing
     │
     ▼
ReLU Activation
     │
     ▼
Energy Update
     │
     ▼
Loss Calculation
     │
     ▼
Weight & Bias Update
```

Milestones 2-4's formal layer (see `docs/ARCHITECTURE.md` for the full equations):

```text
H_out = activation( (A_hat ⊙ Interference(phase) ⊙ decoherence) @ H_in @ W + b )
```

stacked into a `models.quantum_gnn.QuantumGNN`, used either directly for node classification or wrapped in a `GraphClassifier` (readout + linear head) for graph classification.

---

# Visualisation Dashboard

The current prototype generates four visual outputs to aid interpretation and debugging.

## Initial Graph

Visual representation of node energies, phases, and weighted connections before training.


![Initial Graph](docs/initial.png)


## Final Graph

Visual representation of graph state after learning updates.

![Initial Graph](docs/final.png)

## Training Loss

Tracks Mean Squared Error throughout training.

![Initial Graph](docs/loss.png)

## Phase Evolution

Tracks phase changes across all nodes over time.

![Initial Graph](docs/phase.png)

## Milestones 2-4 result plots

Node classification, graph classification, interference-kernel comparison,
and decoherence-rate sweep plots are in `docs/milestone2_node_classification.png`,
`docs/milestone3_graph_classification.png`, and `docs/milestone4_interference_kernels.png` -
see `docs/RESULTS.md` for the numbers behind them.

---

# Research Questions

This project currently investigates the following questions:

### 1. Can phase-dependent interference improve graph message passing?

### 2. Can quantum-inspired state variables create richer node representations?

### 3. How do dynamic phase interactions affect learning stability?

### 4. Can quantum-inspired propagation mechanisms compete with traditional GNN architectures?

### 5. How should quantum-inspired concepts be integrated into graph learning systems?

Milestones 2-4 give a first, honest (synthetic-data) pass at questions 1 and 3: `docs/RESULTS.md` shows phase-modulated message passing clearing random chance by a wide margin on both node and graph classification, and shows decoherence damping giving a small, consistent (not yet statistically validated) improvement. Questions 2, 4 and 5 remain open for Milestone 5+ once the real benchmark datasets are reachable.

---

# Progress & Milestones

## Milestone 1 — Research Prototype

**Status:** ✅ Complete

Implemented:

* [x] Custom graph structure
* [x] Custom node representation
* [x] Energy modelling
* [x] Phase modelling
* [x] Weighted message passing
* [x] Phase interference mechanism
* [x] Forward propagation
* [x] Local learning updates
* [x] Loss tracking
* [x] Dynamic phase evolution
* [x] Visualisation dashboard
* [x] Academic prototype submission

This milestone establishes the first complete working prototype.

---

## Milestone 2 — Formal Graph Neural Network Layer

**Status:** ✅ Complete

* [x] Formal message passing layer (`layers/message_passing.py`)
* [x] Node feature vectors (every node is now a vector, not a scalar)
* [x] Improved learning architecture (real backprop + Adam, `training/`)
* [x] Modular propagation framework (`layers/` + `models/quantum_gnn.py`, shared by both task types)
* [x] Scalable graph support (NumPy/SciPy adjacency matrices, `layers/aggregation.py` - tested up to 500 nodes, see `docs/RESULTS.md`)

---

## Milestone 3 — Real Dataset Integration

**Status:** ✅ Complete (loaders + pipeline written and tested; see caveat below)

Datasets targeted:

* Cora, CiteSeer, PubMed (node classification)
* MUTAG (graph classification)

* [x] Node classification pipeline (`training/trainer.NodeClassifierTrainer`)
* [x] Graph classification pipeline (`training/trainer.GraphClassifierTrainer`)
* [x] Dataset preprocessing (`utils/graph_builder.py` - real-dataset loaders + synthetic generators)
* [x] Performance evaluation (`docs/RESULTS.md`)

> **Caveat:** the real dataset loaders (`load_cora`, `load_citeseer`, `load_pubmed`, `load_mutag`) are written against the datasets' standard public formats but were built in a sandboxed environment that couldn't reach the download hosts. They're untested against the literal files - see `docs/RESULTS.md` for what was actually verified (a structurally-faithful synthetic stand-in) and run them from a machine with normal internet access to confirm against the real data.

---

## Milestone 4 — Advanced Quantum-Inspired Dynamics

**Status:** 🔄 Partially complete

* [x] Complex-valued representations (`layers/quantum_interference.complex_amplitude`, `complex_interference_matrix`)
* [ ] Quantum-walk-inspired propagation *(not started - see docs/ARCHITECTURE.md section 5 for why this needs its own design)*
* [x] Enhanced phase modelling (phase is now trained, not just drifted)
* [x] Decoherence-inspired mechanisms (`layers/quantum_interference.decoherence_factor`)
* [x] Alternative interference functions (squared-cosine, Gaussian)

---

## Milestone 5 — Benchmarking & Evaluation

**Status:** 📋 Planned

Objectives:

* [ ] Compare against standard GNNs
* [ ] Compare convergence behaviour
* [ ] Stability analysis
* [ ] Learning performance evaluation
* [ ] Confirm Milestone 3 results against the real Cora/CiteSeer/PubMed/MUTAG files (currently only verified on structurally-faithful synthetic data - see docs/RESULTS.md)
* [ ] Multi-seed statistical significance test for the Milestone 4 decoherence result

---

## Milestone 6 — Research-Grade Framework

**Status:** 🎯 Long-Term Goal

Objectives:

* [ ] Full Quantum-Inspired GNN architecture
* [ ] Modular framework design
* [ ] Reproducible experiments
* [ ] Research publication preparation
* [ ] Open-source research platform

---

# Roadmap

```text
Milestone 1  ██████████ Complete

Milestone 2  ██████████ Complete

Milestone 3  █████████░ Complete (pending real-data confirmation)

Milestone 4  ██████░░░░ Partial

Milestone 5  ░░░░░░░░░░ Planned

Milestone 6  ░░░░░░░░░░ Planned
```

---

# Installation

## Requirements

* Python 3.12.10

## Install Dependencies

```bash
pip install -r requirements.txt
```

(`numpy`, `scipy`, `networkx`, `matplotlib`, `pytest` - see `requirements.txt`.)

## Run Milestone 1 (original prototype)

```bash
python main.py
```

## Run Milestones 2-4

```bash
python -m pytest tests/ -v          # unit tests + gradient checks
python demo_milestones_2_3_4.py     # trains on synthetic data, writes docs/*.png + docs/milestone_results.json
```

To use the real datasets once you have normal internet access:

```python
from utils.graph_builder import load_cora
data = load_cora()  # downloads + caches into data/planetoid/
```

---

# Repository Structure

```text
quantum-inspired-graph-learning/
│
├── Node.py                     # Milestone 1 (unchanged)
├── Graph.py                    # Milestone 1 (unchanged)
├── main.py                     # Milestone 1 (unchanged)
├── graph_representation.py     # Milestone 1 (unchanged)
├── demo_milestones_2_3_4.py    # Runs everything below, produces docs/*.png + results
├── requirements.txt
├── README.md
│
├── layers/
│   ├── quantum_interference.py # interference kernels, decoherence, complex amplitudes
│   ├── aggregation.py          # adjacency matrix construction + normalisation
│   ├── message_passing.py      # the formal layer (forward + backward)
│   └── update.py               # activations + graph-level readout functions
│
├── models/
│   └── quantum_gnn.py          # QuantumGNN (stacked layers), LinearHead, GraphClassifier
│
├── training/
│   ├── loss.py                 # softmax cross-entropy (+ original MSE)
│   ├── optimizer.py            # SGD, Adam
│   └── trainer.py              # NodeClassifierTrainer, GraphClassifierTrainer
│
├── utils/
│   └── graph_builder.py        # real dataset loaders + synthetic dataset generators
│
├── tests/
│   └── test_layers.py          # unit tests + numerical gradient checks
│
└── docs/
    ├── ARCHITECTURE.md         # full design + maths writeup
    ├── RESULTS.md              # what was run, numbers, honest caveats
    ├── initial.png / final.png / loss.png / phase.png   (Milestone 1)
    └── milestone2_*.png / milestone3_*.png / milestone4_*.png / milestone_results.json
```

---

# Future Research Directions

Potential areas of investigation include:

* Quantum-inspired message passing
* Graph representation learning
* Dynamical graph systems
* Complex-valued neural networks
* Quantum machine learning
* Explainable graph learning
* Physics-inspired artificial intelligence

---

# Author

**Muhammad Azaan Anjam**

Computer Science Student

Newcastle University

GitHub: https://github.com/Azaan78

Copyright (c) 2026 Muhammad Azaan Anjam

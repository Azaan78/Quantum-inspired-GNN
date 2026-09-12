# Architecture: Milestones 2-4

This document explains what changed between Milestone 1 (the original
4-qubit prototype) and the modular framework added for Milestones 2-4, and
the maths behind each piece. It's meant to be read alongside the code in
`layers/`, `models/`, `training/`, and `utils/` - each module's docstring
cross-references the relevant section here.

## 1. From scalars to feature vectors (Milestone 2)

Milestone 1 gave every node exactly one number, `energy`, that played the
role of both "input feature" and "hidden state". Milestone 2 replaces that
with a proper feature vector `h_i ∈ R^F` per node, and a **layer** that maps
a matrix of such vectors, `H ∈ R^(N x F_in)`, to a new matrix `H' ∈ R^(N x
F_out)`:

```
H' = σ( A_q @ H @ W + b )
```

- `W ∈ R^(F_in x F_out)`, `b ∈ R^(F_out)` - ordinary learnable weights, the
  same role Milestone 1's per-edge `weights` dict played, just now shared
  across all nodes and expressed as a matrix multiply instead of a
  dictionary lookup.
- `σ` - an activation function (ReLU between hidden layers, identity on the
  output layer so the final layer produces raw logits for the loss
  function).
- `A_q` - the **quantum-modulated propagation matrix**, described next.

## 2. The quantum-modulated propagation matrix

```
A_q = A_hat ⊙ I(phase) ⊙ decoherence(depth)
```

- `A_hat = D^(-1/2) (A + I_n) D^(-1/2)` - the standard symmetric-normalised
  adjacency matrix (Kipf & Welling, 2017), which keeps aggregated feature
  magnitudes stable regardless of node degree. (`layers/aggregation.py`)
- `I(phase)[i, j] = cos(phase_i - phase_j)` - Milestone 1's interference
  rule, applied to every pair of nodes at once instead of one edge at a
  time. (`layers/quantum_interference.py`)
- `decoherence(depth) = exp(-γ · depth)` - Milestone 4's addition; with
  `γ = 0` (the default) this is 1 everywhere and has no effect.

`⊙` is elementwise (Hadamard) multiplication, so `A_q` has the same sparsity
pattern as the underlying graph (a zero entry in `A_hat` stays zero,
however the nodes' phases happen to line up) - interference can only reduce
or redirect messages that would already flow along a real edge, not create
a new connection out of nothing.

## 3. Phase as a *learnable* parameter, and the non-local learning rule

Milestone 1's phase only ever moved by a small random nudge each epoch
(`node.phase += random.uniform(-0.05, 0.05)`) - it wasn't learned from the
loss at all. One of Milestone 2's listed open items was "a non-local
learning rule"; here's how this codebase answers that.

Because `phase` feeds into `I(phase)`, which feeds into `A_q`, which feeds
into every layer's forward pass, the loss gradient can be backpropagated
all the way through to `phase` itself. The derivative of the interference
term is:

```
∂I[i,j]/∂phase_i = -sin(phase_i - phase_j)
∂I[i,j]/∂phase_j =  sin(phase_i - phase_j)
```

Summing both contributions over every neighbour `j` of `i` (and simplifying
using `sin(phase_j - phase_i) = -sin(phase_i - phase_j)`) gives a clean,
vectorisable expression:

```
∂L/∂phase_i = -Σ_j ( ∂L/∂I[i,j] + ∂L/∂I[j,i] ) · sin(phase_i - phase_j)
```

implemented in `layers/quantum_interference.interference_gradient()`, and
verified against a numerical (finite-difference) gradient in
`tests/test_layers.py::test_message_passing_layer_phase_gradient_matches_numerical`.

**Why this is non-local:** every layer in the stack shares the *same* phase
vector (`models/quantum_gnn.QuantumGNN` passes one `phase` array through
every layer). `QuantumGNN.backward()` therefore sums the phase gradient
across *all* layers before it's applied. With a 2- or 3-layer network, a
node's phase update carries information from errors computed 2-3 hops
away, not just its immediate neighbour's error - unlike Milestone 1's
single-hop, single-layer rule
(`node.weights[neighbour] -= lr * error * neighbour.energy`), which only
ever looked one edge away.

## 4. Milestone 3: node classification vs. graph classification

Two tasks, one shared backbone (`models/quantum_gnn.QuantumGNN`):

- **Node classification** (Cora/CiteSeer/PubMed): one graph, every node
  gets its own label (a paper's topic). The final layer's output width is
  set to `num_classes`, so `QuantumGNN`'s output IS the per-node logits -
  no extra head needed. Only a subset of nodes contribute to the loss at
  any time (`train_mask`/`val_mask`/`test_mask`), even though every node
  still participates in message passing - this is the standard
  "transductive" setup used by the original GCN paper and is why Planetoid
  splits label only ~5% of nodes for training.
- **Graph classification** (MUTAG): many small graphs, one label per graph
  (mutagenic or not). `models/quantum_gnn.GraphClassifier` runs the same
  `QuantumGNN` backbone over each graph, **pools** the resulting node
  embeddings into one vector with a readout function
  (`layers/update.py` - mean/sum/max), then classifies that vector with a
  `LinearHead`.

Both tasks reuse `layers/message_passing.QuantumMessagePassingLayer`
unchanged - this is the "modular propagation framework" Milestone 2 called
for: the same layer, unmodified, is stacked differently depending on the
task.

## 5. Milestone 4 (partial)

Four of the five listed objectives were built; one is left as documented
future work.

| Objective | Status | Where |
|---|---|---|
| Alternative interference functions | Done | `layers/quantum_interference.py`: `squared_cosine_interference`, `gaussian_phase_interference` |
| Decoherence-inspired mechanisms | Done | `layers/quantum_interference.decoherence_factor`, wired into every layer via a `decoherence_rate` parameter |
| Complex-valued representations | Done (forward pass) | `layers/quantum_interference.complex_amplitude`, `complex_interference_matrix` - reduces exactly to the real cosine kernel when every node's magnitude is 1 (tested) |
| Enhanced phase modelling | Done | phase is now a trained parameter (section 3), not just random drift |
| Quantum-walk-inspired propagation | **Not started** | Left for a future milestone - a real quantum walk uses a *unitary* evolution operator, which is a materially different mathematical object from the message-passing formulation used throughout this codebase, and deserves its own design rather than being bolted on |

Only the cosine kernel currently has an analytic (trained) gradient through
`phase`; the squared-cosine and Gaussian kernels are available for
forward-pass experiments (see `demo_milestones_2_3_4.py`) but aren't yet
wired into the trainer's backward pass -
`layers/quantum_interference.interference_gradient()` raises
`NotImplementedError` if you ask for one, rather than silently producing a
wrong gradient.

## 6. What was and wasn't run against real data

This is documented in more detail in `utils/graph_builder.py`'s module
docstring and in `docs/RESULTS.md`, but in short: the sandboxed environment
this was built in couldn't reach the public dataset hosts (GitHub raw
files, chrsmrrs.com) due to network policy, so the real-dataset loaders
(`load_cora`, `load_citeseer`, `load_pubmed`, `load_mutag`) are written
against the datasets' standard formats but have not been run against the
actual files. Everything else - every layer, every gradient, the full
training loop - has been run and verified, against synthetic data built to
have the same structural properties (homophily, sparse categorical
features, a learnable structural motif) as the real benchmarks.

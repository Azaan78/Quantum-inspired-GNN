"""
utils/graph_builder.py
------------------------
Milestone 2's "scalable graph support" (build adjacency/feature matrices
from real edge lists instead of Python object graphs) and Milestone 3's
"dataset preprocessing" for Cora / CiteSeer / PubMed (node classification)
and MUTAG (graph classification).

IMPORTANT - a note on where this was actually run and tested
==============================================================
This file was written and tested from a sandboxed cloud workspace whose
outbound network access is restricted by organisation policy to a small
allowlist (package registries etc.) - it could NOT reach either
github.com/kimiyoung/planetoid (the standard Cora/CiteSeer/PubMed source)
or chrsmrrs.com (the standard MUTAG source) to actually download the real
files. The loaders below are written faithfully against those datasets'
well-documented standard formats, but have only been exercised against the
synthetic-but-structurally-faithful datasets generated further down this
file (see synthetic_citation_graph / synthetic_graph_classification_dataset)
- NOT against the real public benchmarks.

Run load_cora() / load_citeseer() / load_pubmed() / load_mutag() from a
machine with normal internet access (e.g. locally) to fetch and use the
real data; if a URL below has moved, that's the first thing to check.
"""

import os
import pickle
import urllib.request
import zipfile
import io

import numpy as np
import scipy.sparse as sp

from layers.aggregation import normalize_adjacency


# ---------------------------------------------------------------------------
# Real dataset loaders (Milestone 3)
# ---------------------------------------------------------------------------

PLANETOID_BASE_URL = "https://github.com/kimiyoung/planetoid/raw/master/data"
MUTAG_URL = "https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip"


def _download(url, dest_path):
    if os.path.exists(dest_path):
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {url}\n"
            f"(this loader needs normal internet access - see the module "
            f"docstring in utils/graph_builder.py). Original error: {exc}"
        ) from exc
    return dest_path


def _load_planetoid(name, data_dir="data"):
    """
    Loads Cora / CiteSeer / PubMed in the standard format introduced by
    Yang, Cohen & Salakhutdinov (2016) and used by Kipf & Welling's GCN -
    eight small files per dataset (ind.<name>.x/tx/allx/y/ty/ally/graph and
    ind.<name>.test.index), pickled NumPy/SciPy objects plus a plain-text
    index file.

    Returns dict with keys: X (N, F) dense features, A_norm (N, N)
    normalised adjacency, labels (N,) int class ids, train_mask, val_mask,
    test_mask (all (N,) bool). Splits follow the original paper: 20 labelled
    nodes per class for training, 500 for validation, 1000 for testing.
    """
    suffixes = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = {}
    for suffix in suffixes:
        path = _download(
            f"{PLANETOID_BASE_URL}/ind.{name}.{suffix}",
            os.path.join(data_dir, "planetoid", f"ind.{name}.{suffix}"),
        )
        with open(path, "rb") as f:
            objects[suffix] = pickle.load(f, encoding="latin1")

    test_idx_path = _download(
        f"{PLANETOID_BASE_URL}/ind.{name}.test.index",
        os.path.join(data_dir, "planetoid", f"ind.{name}.test.index"),
    )
    test_idx = np.loadtxt(test_idx_path, dtype=int)
    test_idx_sorted = np.sort(test_idx)

    allx, ally = objects["allx"], objects["ally"]
    tx, ty = objects["tx"], objects["ty"]
    graph = objects["graph"]  # dict: node_id -> list of neighbour ids

    # PubMed/CiteSeer have a handful of isolated test nodes that need
    # slotting into the right (sorted) position - the classic Planetoid
    # loading quirk. See the original `gcn` repo's `utils.py` for context.
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx, :] = features[test_idx_sorted, :]
    X = np.asarray(features.todense(), dtype=np.float64)

    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[test_idx_sorted, :]
    labels = labels.argmax(axis=1)

    num_nodes = X.shape[0]
    edge_list = []
    for node, neighbours in graph.items():
        for nb in neighbours:
            edge_list.append((node, nb))
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i, j in edge_list:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A_norm = normalize_adjacency(A)

    idx_train = np.arange(len(objects["y"]))            # first 20*num_classes nodes
    idx_val = np.arange(len(objects["y"]), len(objects["y"]) + 500)
    idx_test = test_idx_sorted

    def mask_from_idx(idx):
        m = np.zeros(num_nodes, dtype=bool)
        m[idx] = True
        return m

    return {
        "X": X,
        "A_norm": A_norm,
        "labels": labels,
        "train_mask": mask_from_idx(idx_train),
        "val_mask": mask_from_idx(idx_val),
        "test_mask": mask_from_idx(idx_test),
    }


def load_cora(data_dir="data"):
    return _load_planetoid("cora", data_dir)


def load_citeseer(data_dir="data"):
    return _load_planetoid("citeseer", data_dir)


def load_pubmed(data_dir="data"):
    return _load_planetoid("pubmed", data_dir)


def load_mutag(data_dir="data"):
    """
    Loads MUTAG in the standard TU Dortmund graph-benchmark format: a small
    zip containing four text files (edges, a graph-id-per-node indicator,
    per-graph labels, and per-node categorical labels).

    Returns a list of {"X": (n_i, F), "A_norm": (n_i, n_i), "label": int}
    dicts, one per graph, ready for training.trainer.GraphClassifierTrainer.
    """
    zip_path = _download(MUTAG_URL, os.path.join(data_dir, "MUTAG.zip"))
    extract_dir = os.path.join(data_dir, "MUTAG")
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)

    prefix = os.path.join(extract_dir, "MUTAG")

    def read_ints(suffix):
        with open(f"{prefix}_{suffix}.txt") as f:
            return [int(line.strip()) for line in f if line.strip()]

    def read_edges():
        with open(f"{prefix}_A.txt") as f:
            edges = []
            for line in f:
                if not line.strip():
                    continue
                a, b = line.strip().split(",")
                edges.append((int(a), int(b)))
            return edges

    graph_indicator = np.array(read_ints("graph_indicator"))  # 1-indexed graph id per (global) node
    graph_labels = np.array(read_ints("graph_labels"))
    node_labels = np.array(read_ints("node_labels"))
    edges = read_edges()  # 1-indexed global node ids

    num_node_label_classes = node_labels.max() + 1
    num_graphs = graph_indicator.max()

    dataset = []
    label_map = {val: i for i, val in enumerate(sorted(set(graph_labels.tolist())))}

    for gid in range(1, num_graphs + 1):
        global_ids = np.where(graph_indicator == gid)[0] + 1  # back to 1-indexed
        local_index = {g: local for local, g in enumerate(global_ids)}
        n = len(global_ids)

        X = np.zeros((n, num_node_label_classes), dtype=np.float64)
        for g in global_ids:
            X[local_index[g], node_labels[g - 1]] = 1.0

        A = np.zeros((n, n), dtype=np.float64)
        for a, b in edges:
            if a in local_index and b in local_index:
                A[local_index[a], local_index[b]] = 1.0
                A[local_index[b], local_index[a]] = 1.0
        A_norm = normalize_adjacency(A)

        label = label_map[graph_labels[gid - 1]]
        dataset.append({"X": X, "A_norm": A_norm, "label": label})

    return dataset


# ---------------------------------------------------------------------------
# Synthetic, network-free datasets - used by tests/test_layers.py and by the
# demo run in this milestone's docs/RESULTS.md, since the real datasets
# above couldn't be downloaded from this sandbox (see module docstring).
# Structurally faithful to the real thing: homophilic community structure
# and sparse categorical features for node classification (like Cora),
# small labelled graphs distinguished by a substructure for graph
# classification (like MUTAG) - NOT just random noise, so a model that
# genuinely uses graph structure will outperform one that ignores it.
# ---------------------------------------------------------------------------

def synthetic_citation_graph(num_nodes=300, num_features=64, num_classes=4,
                              homophily=0.85, avg_degree=4, seed=0):
    """
    Generates a Cora-shaped synthetic dataset: every node has a class, edges
    are far more likely between same-class nodes ("homophily", the property
    real citation graphs have - papers cite similar papers) than between
    different classes, and each class has a characteristic subset of "hot"
    sparse binary features (like a bag-of-words vocabulary a topic tends to
    use).
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=num_nodes)

    # ---- edges: stochastic-block-model-style, tuned for homophily ----
    edge_list = []
    target_edges = int(num_nodes * avg_degree / 2)
    same_class_nodes = {c: np.where(labels == c)[0] for c in range(num_classes)}

    attempts = 0
    seen = set()
    while len(edge_list) < target_edges and attempts < target_edges * 50:
        attempts += 1
        i = rng.integers(0, num_nodes)
        if rng.random() < homophily and len(same_class_nodes[labels[i]]) > 1:
            j = rng.choice(same_class_nodes[labels[i]])
        else:
            j = rng.integers(0, num_nodes)
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        edge_list.append(key)

    # ---- features: each class has its own characteristic "hot" indices ----
    features_per_class = max(4, num_features // (num_classes * 2))
    class_feature_sets = {
        c: rng.choice(num_features, size=features_per_class, replace=False)
        for c in range(num_classes)
    }
    X = (rng.random((num_nodes, num_features)) < 0.02).astype(np.float64)  # background sparsity
    for i in range(num_nodes):
        hot = class_feature_sets[labels[i]]
        turn_on = rng.random(len(hot)) < 0.6
        X[i, hot[turn_on]] = 1.0

    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i, j in edge_list:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A_norm = normalize_adjacency(A)

    return {
        "X": X,
        "A_norm": A_norm,
        "labels": labels,
        **train_val_test_masks(labels, num_classes, seed=seed),
    }


def train_val_test_masks(labels, num_classes, per_class_train=20, val_size=0.2, seed=0):
    """
    Planetoid-style split: a fixed small number of labelled nodes per class
    for training (mirrors the real Cora/CiteSeer/PubMed splits used above),
    a validation slice, and everything else held out for test.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    train_idx = []
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        train_idx.extend(class_idx[:per_class_train].tolist())
    train_idx = np.array(train_idx)

    remaining = np.array([i for i in range(n) if i not in set(train_idx.tolist())])
    rng.shuffle(remaining)
    n_val = int(len(remaining) * val_size)
    val_idx = remaining[:n_val]
    test_idx = remaining[n_val:]

    def mask(idx):
        m = np.zeros(n, dtype=bool)
        m[idx] = True
        return m

    return {"train_mask": mask(train_idx), "val_mask": mask(val_idx), "test_mask": mask(test_idx)}


def synthetic_graph_classification_dataset(num_graphs=150, min_nodes=10, max_nodes=25,
                                            num_node_labels=7, seed=0):
    """
    Generates a MUTAG-shaped dataset: many small graphs, each with
    categorical (one-hot) node labels, and a binary graph-level label that
    depends on a local STRUCTURAL MOTIF - whether the graph contains an
    edge directly connecting two "special" (label-0) atoms - rather than
    just node-label frequencies. That mirrors how MUTAG's label actually
    works (mutagenicity is driven by specific bonded substructures, e.g.
    particular functional groups), and, importantly, is a motif ordinary
    message passing CAN learn to detect (a node aggregating its neighbours'
    labels directly sees a label-0 neighbour). This is deliberately NOT
    "contains a triangle": counting/detecting triangles is a well-known
    theoretical blind spot of standard message-passing GNNs (they're only
    as powerful as the 1-dimensional Weisfeiler-Leman graph isomorphism
    test, which provably cannot count triangles - see Xu et al., "How
    Powerful are Graph Neural Networks?", 2019) - a task like that would
    make the model look broken when actually it's a known, published
    architectural limit, not a bug in this implementation.
    """
    rng = np.random.default_rng(seed)
    dataset = []
    special_label = 0

    for _ in range(num_graphs):
        n = rng.integers(min_nodes, max_nodes + 1)
        has_special_bond = rng.random() < 0.5

        node_labels = rng.integers(1, num_node_labels, size=n)  # avoid label 0 by default
        if has_special_bond:
            # Guarantee at least one label-0/label-0 pair exists somewhere.
            i, j = rng.choice(n, size=2, replace=False)
            node_labels[i] = special_label
            node_labels[j] = special_label
        X = np.zeros((n, num_node_labels), dtype=np.float64)
        X[np.arange(n), node_labels] = 1.0

        # Random connected base structure (spanning tree) plus a little
        # extra density, independent of the label assignment above.
        A = np.zeros((n, n), dtype=np.float64)
        for i in range(1, n):
            j = rng.integers(0, i)
            A[i, j] = A[j, i] = 1.0
        extra_edges = rng.integers(0, max(1, n // 4))
        for _ in range(extra_edges):
            i, j = rng.choice(n, size=2, replace=False)
            A[i, j] = A[j, i] = 1.0

        if has_special_bond:
            # Make sure the two special atoms are actually bonded to each
            # other (not just both present somewhere in the molecule).
            A[i, j] = A[j, i] = 1.0
        else:
            # Make sure no two special-labelled nodes ended up bonded by
            # chance from the random edges above.
            special_nodes = np.where(node_labels == special_label)[0]
            for a in special_nodes:
                for b in special_nodes:
                    if a != b:
                        A[a, b] = A[b, a] = 0.0

        label = int(has_special_bond)
        A_norm = normalize_adjacency(A)
        dataset.append({"X": X, "A_norm": A_norm, "label": label})

    return dataset

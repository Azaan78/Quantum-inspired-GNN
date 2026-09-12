"""
training/loss.py
-----------------
Milestone 1 only had Mean Squared Error, used against a hand-crafted
"expected future state" (there were no real labels - `generate_true_future`
just averaged a node's own and its neighbours' energy). That was a
reasonable way to demonstrate *that* the network could learn something, but
it isn't how you train a real classifier.

Milestone 3 needs real supervised learning against real labels (Cora's
paper topics, MUTAG's mutagenicity label), so this module adds a proper
softmax + cross-entropy loss, with an analytic gradient, while keeping MSE
around for backward compatibility / anyone who wants to reproduce the
Milestone 1 style of training.
"""

import numpy as np


def mse_loss(y_pred, y_true):
    """Milestone 1's loss, kept as-is for reference/backward compatibility."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(np.mean((y_pred - y_true) ** 2))


def mse_grad(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    return 2.0 * (y_pred - y_true) / y_pred.size


def softmax(logits):
    """Numerically stable softmax over the last axis."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def softmax_cross_entropy(logits, labels, mask=None):
    """
    logits: (N, num_classes) raw scores (node classification) or
            (num_classes,) for a single graph (graph classification, called
            with logits[None, :] by the trainer).
    labels: (N,) integer class indices.
    mask: optional boolean (N,) array - used for node classification, where
          only some nodes (the train/val/test split) contribute to the loss
          even though every node participates in message passing. This is
          exactly how Cora/CiteSeer/PubMed are normally trained: the whole
          graph is used for propagation, but the loss (and its gradient)
          only "sees" the labelled training nodes.

    Returns (loss, dL_dlogits) so the trainer can feed dL_dlogits straight
    into the model's backward() call.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    probs = softmax(logits)

    n = logits.shape[0]
    if mask is None:
        mask = np.ones(n, dtype=bool)

    idx = np.arange(n)
    correct_class_probs = probs[idx, labels]
    # Avoid log(0) for numerical safety.
    eps = 1e-12
    per_node_loss = -np.log(np.clip(correct_class_probs, eps, 1.0))

    masked_loss = per_node_loss[mask]
    loss = float(masked_loss.mean()) if masked_loss.size > 0 else 0.0

    # Gradient of softmax + cross-entropy w.r.t. logits: (probs - one_hot),
    # zeroed out and re-scaled for unmasked nodes so masked nodes get no
    # gradient at all (they still had a forward pass, just no loss).
    one_hot = np.zeros_like(probs)
    one_hot[idx, labels] = 1.0
    grad = (probs - one_hot)
    num_active = mask.sum() if mask.sum() > 0 else 1
    grad = np.where(mask[:, None], grad / num_active, 0.0)

    return loss, grad


def accuracy(logits, labels, mask=None):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    preds = logits.argmax(axis=-1)
    n = logits.shape[0]
    if mask is None:
        mask = np.ones(n, dtype=bool)
    if mask.sum() == 0:
        return 0.0
    return float((preds[mask] == labels[mask]).mean())

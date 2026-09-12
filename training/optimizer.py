"""
training/optimizer.py
----------------------
Milestone 1's "optimizer" was one line embedded directly in
back_propagate(): `node.weights[neighbour] -= learning_rate * gradient`
(plain SGD, hard-coded). This module formalises that into swappable
optimizer objects, and adds Adam, which converges far more reliably on a
real multi-class problem like Cora than plain SGD does - important once
we're not just demonstrating "loss goes down" on a toy 4-node graph but
actually trying to hit a respectable classification accuracy.
"""

import numpy as np


class SGD:
    """Milestone 1's update rule, generalised to arrays: param -= lr * grad."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, key, grad):
        # `key` is unused by plain SGD (no per-parameter state to look up),
        # but is accepted so SGD and Adam share one interface - see
        # training/trainer.py, which calls `optimizer.step(key, grad)`
        # without needing to know which optimizer it was given.
        return -self.lr * grad


class Adam:
    """
    Standard Adam optimizer (Kingma & Ba, 2014). Keeps a running estimate of
    the first and second moment of each parameter's gradient, which lets it
    use an effectively adaptive, per-parameter learning rate - this is why
    it's the default choice for training most modern neural networks
    (including GNNs) rather than plain SGD.

    Because our parameters live in several different places (each layer's W
    and b, and the shared phase vector), we key the internal moment
    estimates by whatever string name the caller uses (see
    training/trainer.py), rather than assuming a fixed parameter list.
    """

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = {}
        self._v = {}
        self._t = {}

    def step(self, key, grad):
        if key not in self._m:
            self._m[key] = np.zeros_like(grad)
            self._v[key] = np.zeros_like(grad)
            self._t[key] = 0

        self._t[key] += 1
        t = self._t[key]
        m = self.beta1 * self._m[key] + (1 - self.beta1) * grad
        v = self.beta2 * self._v[key] + (1 - self.beta2) * (grad ** 2)
        self._m[key] = m
        self._v[key] = v

        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)

        return -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

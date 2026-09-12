"""
quantum_interference.py
------------------------
Milestone 2 + Milestone 4.

This module is the mathematical heart of the "quantum-inspired" part of the
project. In Milestone 1, every node carried a single scalar phase, and
interference between two connected nodes was just:

    interference = cos(phase_i - phase_j)

That idea is kept here, but it's now formalised, vectorised (so it works on
graphs with thousands of nodes, not just 4), and extended with the
Milestone 4 items:

    * alternative interference kernels (not just cosine)
    * a decoherence-inspired damping term (interference weakens with depth/time,
      like a quantum system losing coherence)
    * a complex-valued representation (amplitude * phase, instead of a bare
      real phase), which is the more physically faithful way to describe
      interference between two quantum amplitudes

Everything here is pure NumPy so it can run without a deep learning
framework, matching the "build it yourself" spirit of Milestone 1.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Milestone 2: the original cosine interference, vectorised over a whole graph
# ---------------------------------------------------------------------------

def phase_difference_matrix(phase):
    """
    Given a phase vector of shape (N,), return the NxN matrix of pairwise
    phase differences: diff[i, j] = phase[i] - phase[j].

    This is the vectorised version of Milestone 1's
        phase_difference = node.phase - neighbour.phase
    computed once for the whole graph instead of one pair at a time.
    """
    phase = np.asarray(phase, dtype=np.float64)
    return phase[:, None] - phase[None, :]


def cosine_interference(phase):
    """
    Milestone 1's interference rule, generalised to the whole graph:
        I[i, j] = cos(phase_i - phase_j)

    Returns an NxN matrix. I[i, i] = 1 (a node is always perfectly "in phase"
    with itself), and I is symmetric because cos(-x) = cos(x).
    """
    diff = phase_difference_matrix(phase)
    return np.cos(diff)


def squared_cosine_interference(phase):
    """
    Milestone 4: an alternative interference kernel.

    Using cos^2 instead of cos is the classic "Born rule" shape you get when
    you interfere two probability amplitudes and look at the resulting
    intensity/probability (which must be >= 0) rather than a raw amplitude.
    Unlike plain cosine, this kernel never produces a *negative* interference
    weight - two nodes are always either "constructively" reinforcing
    (I close to 1) or "destructively" cancelling (I close to 0), never
    flipping the sign of the connection outright.
    """
    diff = phase_difference_matrix(phase)
    return np.cos(diff) ** 2


def gaussian_phase_interference(phase, sigma=1.0):
    """
    Milestone 4: a second alternative interference kernel.

    Instead of the periodic cosine, this uses a Gaussian bump centred on
    phase_i == phase_j: nodes that are exactly in phase interfere at full
    strength (I == 1), and the interference falls off smoothly (and never
    goes negative) as the phase difference grows, controlled by `sigma`.
    This is a common alternative to a periodic kernel when you want
    "closeness in phase" to behave like a distance rather than an angle.
    """
    diff = phase_difference_matrix(phase)
    return np.exp(-(diff ** 2) / (2.0 * sigma ** 2))


INTERFERENCE_KERNELS = {
    "cosine": cosine_interference,
    "squared_cosine": squared_cosine_interference,
    "gaussian": gaussian_phase_interference,
}


def interference_matrix(phase, kernel="cosine", **kernel_kwargs):
    """
    Dispatch to one of the interference kernels above by name. This is what
    layers/message_passing.py calls, so swapping Milestone 4's "alternative
    interference functions" in and out is a one-word config change rather
    than a code change.
    """
    if kernel not in INTERFERENCE_KERNELS:
        raise ValueError(
            f"Unknown interference kernel '{kernel}'. "
            f"Choose from {list(INTERFERENCE_KERNELS)}."
        )
    return INTERFERENCE_KERNELS[kernel](phase, **kernel_kwargs)


def interference_gradient(phase, dL_dI, kernel="cosine", **kernel_kwargs):
    """
    Backpropagate a gradient through the interference matrix back to the
    phase vector.

    This is what makes phase a *learnable* parameter rather than something
    that only ever drifts randomly (as it did in Milestone 1). Because the
    interference matrix feeds into every message-passing layer, and layers
    are stacked, the gradient a phase receives here is a sum over every
    node's error signal that phase could have influenced, possibly several
    hops away - i.e. a genuinely *non-local* learning rule for phase, not
    just "look at your immediate neighbour's error" like Milestone 1's
    weight/bias update.

    Only the cosine kernel's analytic gradient is implemented (it's the
    kernel actually used during training below); the others are provided
    for experimentation/visualisation, matching the project's "formal
    layer, but still exploratory" status.
    """
    if kernel != "cosine":
        raise NotImplementedError(
            "Analytic gradients are currently only implemented for the "
            "'cosine' kernel. Squared-cosine/Gaussian are available for "
            "forward-pass experiments (see main.py-style visualisation "
            "scripts) but are not yet wired into the trainer."
        )

    diff = phase_difference_matrix(phase)
    sin_diff = np.sin(diff)  # S[i, j] = sin(phase_i - phase_j)

    # d(I[i, j]) / d(phase_i) = -sin(phase_i - phase_j)
    # d(I[i, j]) / d(phase_j) =  sin(phase_i - phase_j)
    #
    # Summing both contributions for every node gives (see docs/ARCHITECTURE.md
    # for the full derivation):
    #   grad_phase = -sum_j (dL_dI[i,j] + dL_dI[j,i]) * sin_diff[i,j]
    symmetrised = dL_dI + dL_dI.T
    grad_phase = -np.sum(symmetrised * sin_diff, axis=1)
    return grad_phase


# ---------------------------------------------------------------------------
# Milestone 4: decoherence-inspired damping
# ---------------------------------------------------------------------------

def decoherence_factor(depth, decoherence_rate=0.0):
    """
    A real quantum system loses coherence over time/distance - interference
    effects fade and the system starts behaving classically. We model that
    here with a simple exponential damping factor applied to the
    interference matrix as a function of "depth" (how many message-passing
    layers/hops the signal has travelled):

        factor = exp(-decoherence_rate * depth)

    With decoherence_rate = 0 this is exactly 1.0 everywhere, i.e. no
    decoherence, which reproduces Milestone 2/3 behaviour unchanged - so
    this is an *opt-in* extension, not a change to earlier milestones.
    """
    return float(np.exp(-decoherence_rate * depth))


# ---------------------------------------------------------------------------
# Milestone 4: complex-valued node representation
# ---------------------------------------------------------------------------

def complex_amplitude(magnitude, phase):
    """
    Represent each node as a complex amplitude z = r * e^{i * phase}, where
    `magnitude` (r) plays the role Milestone 1's scalar `energy` used to
    play, and `phase` is the same phase variable as before. This is the
    textbook way quantum amplitudes are written, and it makes "interference"
    a literal property of complex-number multiplication rather than an
    analogy bolted on with cos().
    """
    magnitude = np.asarray(magnitude, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    return magnitude * np.exp(1j * phase)


def complex_interference_matrix(magnitude, phase):
    """
    True quantum-style interference between two complex amplitudes z_i, z_j:

        I[i, j] = Re(z_i * conj(z_j)) = r_i * r_j * cos(phase_i - phase_j)

    This reduces to plain cosine_interference() when every magnitude is 1,
    so it's a strict generalisation of the Milestone 2 kernel rather than a
    separate mechanism. The magnitude term additionally lets *how much
    amplitude a node carries* (not just its phase) shape how strongly it
    interferes with its neighbours - closer to how interference actually
    works for quantum amplitudes than a phase-only model.
    """
    z = complex_amplitude(magnitude, phase)
    return np.real(z[:, None] * np.conj(z[None, :]))

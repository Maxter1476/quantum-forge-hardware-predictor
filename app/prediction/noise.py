"""Noise channel primitives operating on classical output distributions.

The prediction engine works in distribution space: each hardware effect is a
stochastic transformation of the probability vector over measured bitstrings.
Distributions are dense numpy vectors of length 2**m where m is the number of
measured qubits; bit k of the string corresponds to tensor axis k.
"""
from __future__ import annotations

import numpy as np


def dist_to_vector(probs: dict[str, float], num_bits: int) -> np.ndarray:
    vec = np.zeros(2**num_bits)
    for bits, p in probs.items():
        vec[int(bits, 2)] = p
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def vector_to_dist(vec: np.ndarray, num_bits: int, tol: float = 1e-9) -> dict[str, float]:
    out = {}
    for idx, p in enumerate(vec):
        if p > tol:
            out[format(idx, f"0{num_bits}b")] = float(p)
    return dict(sorted(out.items()))


def total_variation_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def apply_bit_channel(vec: np.ndarray, num_bits: int, axis: int, m: np.ndarray) -> np.ndarray:
    """Apply a column-stochastic 2x2 matrix ``m[out, in]`` to one bit."""
    tensor = vec.reshape((2,) * num_bits)
    tensor = np.tensordot(m, tensor, axes=([1], [axis]))
    tensor = np.moveaxis(tensor, 0, axis)
    return tensor.reshape(-1)


def mix_with_uniform(vec: np.ndarray, weight: float) -> np.ndarray:
    """Depolarizing-style mixing: keep (1-w) of the signal, spread w uniformly."""
    weight = float(np.clip(weight, 0.0, 1.0))
    uniform = np.full_like(vec, 1.0 / len(vec))
    return (1.0 - weight) * vec + weight * uniform


def readout_confusion(p_flip_0to1: float, p_flip_1to0: float) -> np.ndarray:
    """Column-stochastic readout confusion matrix for one qubit."""
    return np.array(
        [
            [1.0 - p_flip_0to1, p_flip_1to0],
            [p_flip_0to1, 1.0 - p_flip_1to0],
        ]
    )


def relaxation_channel(p_relax: float) -> np.ndarray:
    """T1 relaxation in distribution space: |1> decays to |0> with p_relax."""
    return np.array([[1.0, p_relax], [0.0, 1.0 - p_relax]])


def accumulate_fidelity(errors: list[float]) -> float:
    """Probability that *no* gate in the list failed."""
    f = 1.0
    for e in errors:
        f *= max(0.0, 1.0 - e)
    return f


def confidence_intervals(
    probs: dict[str, float],
    shots: int,
    widen_factor: float = 1.0,
    z: float = 1.96,
) -> dict[str, tuple[float, float]]:
    """Normal-approximation CI for each bitstring probability.

    ``widen_factor`` >= 1 inflates intervals to reflect calibration drift.
    """
    out: dict[str, tuple[float, float]] = {}
    for bits, p in probs.items():
        half = z * widen_factor * float(np.sqrt(max(p * (1.0 - p), 1e-12) / max(shots, 1)))
        out[bits] = (max(0.0, p - half), min(1.0, p + half))
    return out


def sample_counts(
    probs: dict[str, float], shots: int, seed: int | None = None
) -> dict[str, int]:
    """Draw multinomial counts from the predicted distribution."""
    rng = np.random.default_rng(seed)
    keys = list(probs)
    p = np.array([probs[k] for k in keys])
    p = p / p.sum()
    draws = rng.multinomial(shots, p)
    return {k: int(c) for k, c in zip(keys, draws) if c > 0}

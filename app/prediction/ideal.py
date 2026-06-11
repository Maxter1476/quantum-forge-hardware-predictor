"""Minimal ideal statevector engine.

Provides the noiseless baseline distribution that the hardware prediction
engine then distorts. Supports up to ``MAX_QUBITS`` qubits.

Bitstring convention: qubit 0 is the *leftmost* character, so for a 2-qubit
register the string "10" means q0=1, q1=0.
"""
from __future__ import annotations

import math

import numpy as np

from app.circuits.ir import CircuitIR, Instruction

MAX_QUBITS = 12

_SQ = 1.0 / math.sqrt(2.0)

_FIXED_GATES: dict[str, np.ndarray] = {
    "i": np.eye(2, dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "h": np.array([[_SQ, _SQ], [_SQ, -_SQ]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "t": np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex),
}


def _rotation(name: str, theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    if name == "rx":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if name == "ry":
        return np.array([[c, -s], [s, c]], dtype=complex)
    if name == "rz":
        return np.array(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
        )
    raise ValueError(f"unknown rotation {name}")


def _single_qubit_matrix(ins: Instruction) -> np.ndarray:
    if ins.name in _FIXED_GATES:
        return _FIXED_GATES[ins.name]
    return _rotation(ins.name, ins.params[0])


class IdealEngine:
    """Statevector simulator for small circuits."""

    def __init__(self, max_qubits: int = MAX_QUBITS):
        self.max_qubits = max_qubits

    def statevector(self, circuit: CircuitIR) -> np.ndarray:
        n = circuit.num_qubits
        if n > self.max_qubits:
            raise ValueError(
                f"circuit has {n} qubits; ideal engine supports up to {self.max_qubits}"
            )
        # state as an n-dimensional tensor, axis k = qubit k
        state = np.zeros((2,) * n, dtype=complex)
        state[(0,) * n] = 1.0

        for ins in circuit.instructions:
            if ins.is_measurement:
                continue
            if ins.is_two_qubit:
                state = self._apply_two_qubit(state, ins)
            else:
                u = _single_qubit_matrix(ins)
                state = np.moveaxis(
                    np.tensordot(u, state, axes=([1], [ins.qubits[0]])),
                    0,
                    ins.qubits[0],
                )
        return state.reshape(-1)

    @staticmethod
    def _apply_two_qubit(state: np.ndarray, ins: Instruction) -> np.ndarray:
        a, b = ins.qubits
        if ins.name == "cx":
            u4 = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
            )
        elif ins.name == "cz":
            u4 = np.diag([1, 1, 1, -1]).astype(complex)
        elif ins.name == "swap":
            u4 = np.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
            )
        else:
            raise ValueError(f"unknown two-qubit gate {ins.name}")
        u = u4.reshape(2, 2, 2, 2)
        state = np.tensordot(u, state, axes=([2, 3], [a, b]))
        return np.moveaxis(state, [0, 1], [a, b])

    def probabilities(self, circuit: CircuitIR) -> dict[str, float]:
        """Ideal probability of each bitstring over all qubits.

        Keys are bitstrings with qubit 0 leftmost; near-zero entries are
        dropped for compactness.
        """
        psi = self.statevector(circuit)
        n = circuit.num_qubits
        probs = np.abs(psi) ** 2
        out: dict[str, float] = {}
        for idx, p in enumerate(probs):
            if p > 1e-12:
                out[format(idx, f"0{n}b")] = float(p)
        return out

    def measured_probabilities(self, circuit: CircuitIR) -> dict[str, float]:
        """Probabilities marginalized onto the measured qubits.

        If the circuit has no explicit measurements, all qubits are assumed
        measured. Bit order in the key follows ascending measured-qubit index.
        """
        measured = circuit.measured_qubits or list(range(circuit.num_qubits))
        measured = sorted(measured)
        full = self.probabilities(circuit)
        out: dict[str, float] = {}
        for bits, p in full.items():
            key = "".join(bits[q] for q in measured)
            out[key] = out.get(key, 0.0) + p
        return dict(sorted(out.items()))

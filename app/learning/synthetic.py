"""Synthetic "hardware run" generation for training the correction layer.

The trick: build a *hidden* version of each hardware profile whose noise
parameters are randomly perturbed, run the prediction engine against it to
play the role of the real device, and sample shot-noisy counts. The visible
profile (what the predictor sees) differs from the hidden truth, so a learned
model can capture the systematic residual.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from app.circuits.ir import CircuitIR, Instruction
from app.hardware.profile import HardwareProfile
from app.prediction.engine import HardwarePredictionEngine
from app.prediction.noise import dist_to_vector

_1Q_GATES = ["x", "h", "s", "t", "rx", "ry", "rz"]


@dataclass
class SyntheticExample:
    """One training example: a circuit, the visible profile, and 'real' counts."""

    circuit: CircuitIR
    profile: HardwareProfile
    true_counts: dict[str, int]
    true_probabilities: dict[str, float]
    hidden_scale: float = 1.0
    metadata: dict = field(default_factory=dict)


def random_circuit(
    rng: random.Random, num_qubits: int, depth: int, name: str = "rand"
) -> CircuitIR:
    """Generate a random circuit measuring all qubits."""
    instructions: list[Instruction] = []
    for _ in range(depth):
        q = rng.randrange(num_qubits)
        if num_qubits >= 2 and rng.random() < 0.35:
            partner = rng.choice([p for p in range(num_qubits) if p != q])
            gate = rng.choice(["cx", "cz"])
            instructions.append(Instruction(name=gate, qubits=[q, partner]))
        else:
            gate = rng.choice(_1Q_GATES)
            params = [rng.uniform(0, 2 * np.pi)] if gate in {"rx", "ry", "rz"} else []
            instructions.append(Instruction(name=gate, qubits=[q], params=params))
    for q in range(num_qubits):
        instructions.append(Instruction(name="measure", qubits=[q], clbits=[q]))
    circuit = CircuitIR(
        name=name, num_qubits=num_qubits, num_clbits=num_qubits, instructions=instructions
    )
    circuit.validate_structure()
    return circuit


def perturb_profile(
    profile: HardwareProfile, rng: random.Random, scale_range: tuple[float, float] = (0.6, 1.8)
) -> tuple[HardwareProfile, float]:
    """Return a hidden-truth profile with all error rates scaled by one factor."""
    scale = rng.uniform(*scale_range)
    hidden = profile.model_copy(deep=True)
    hidden.single_qubit_error = {
        q: min(0.4, e * scale) for q, e in hidden.single_qubit_error.items()
    }
    hidden.two_qubit_error = {
        k: min(0.4, e * scale) for k, e in hidden.two_qubit_error.items()
    }
    hidden.readout_error = {
        q: min(0.4, e * scale) for q, e in hidden.readout_error.items()
    }
    hidden.t1_us = {q: t / scale for q, t in hidden.t1_us.items()}
    hidden.t2_us = {q: t / scale for q, t in hidden.t2_us.items()}
    return hidden, scale


def generate_dataset(
    profile: HardwareProfile,
    num_examples: int = 60,
    shots: int = 1024,
    seed: int = 0,
    max_qubits: int = 4,
) -> list[SyntheticExample]:
    """Generate synthetic hardware-run examples against a visible profile."""
    rng = random.Random(seed)
    engine = HardwarePredictionEngine()
    examples: list[SyntheticExample] = []
    for i in range(num_examples):
        n = rng.randint(2, min(max_qubits, profile.num_qubits))
        depth = rng.randint(3, 10)
        circuit = random_circuit(rng, n, depth, name=f"synthetic-{i}")
        hidden, scale = perturb_profile(profile, rng)
        truth = engine.predict(circuit, hidden, shots=shots, seed=rng.randrange(2**31))
        examples.append(
            SyntheticExample(
                circuit=circuit,
                profile=profile,
                true_counts=truth.predicted_counts,
                true_probabilities=truth.predicted_probabilities,
                hidden_scale=scale,
                metadata={"depth": depth, "num_qubits": n},
            )
        )
    return examples


def empirical_distribution(counts: dict[str, int], num_bits: int) -> np.ndarray:
    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}
    return dist_to_vector(probs, num_bits)

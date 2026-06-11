"""Feature extraction for the learning correction layer."""
from __future__ import annotations

import numpy as np

from app.circuits.ir import CircuitIR
from app.hardware.profile import HardwareProfile
from app.prediction.transpile import MappingEstimate

FEATURE_NAMES = [
    "num_qubits",
    "depth",
    "gate_count",
    "two_qubit_gate_count",
    "added_swap_count",
    "topology_penalty",
    "avg_single_qubit_error",
    "avg_two_qubit_error",
    "avg_readout_error",
    "avg_t1_us",
    "avg_t2_us",
    "calibration_age_hours",
    "two_qubit_density",
]


def extract_features(
    circuit: CircuitIR, profile: HardwareProfile, mapping: MappingEstimate
) -> np.ndarray:
    """Build a fixed-length numeric feature vector for one prediction case."""
    t1_vals = list(profile.t1_us.values())
    t2_vals = list(profile.t2_us.values())
    gate_count = max(circuit.gate_count, 1)
    return np.array(
        [
            circuit.num_qubits,
            circuit.depth,
            circuit.gate_count,
            circuit.two_qubit_gate_count,
            mapping.added_swap_count,
            mapping.topology_penalty,
            profile.avg_single_qubit_error(),
            profile.avg_two_qubit_error(),
            profile.avg_readout_error(),
            sum(t1_vals) / len(t1_vals),
            sum(t2_vals) / len(t2_vals),
            profile.calibration_age_hours(),
            circuit.two_qubit_gate_count / gate_count,
        ],
        dtype=float,
    )

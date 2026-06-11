"""Hardware output prediction engine (the core deliverable).

Transforms the ideal output distribution of a circuit into the distribution a
real (mock) quantum processor would be expected to produce, accounting for
gate error, readout error, decoherence, crosstalk, leakage, calibration
drift, and shot noise — with per-source error accounting.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from app.circuits.ir import CircuitIR
from app.hardware.profile import HardwareProfile
from app.prediction.ideal import IdealEngine
from app.prediction.noise import (
    accumulate_fidelity,
    apply_bit_channel,
    confidence_intervals,
    dist_to_vector,
    mix_with_uniform,
    readout_confusion,
    relaxation_channel,
    sample_counts,
    total_variation_distance,
    vector_to_dist,
)
from app.prediction.transpile import MappingEstimate, estimate_mapping

CROSSTALK_RATE_PER_EVENT = 0.004
LEAKAGE_RATE_PER_TWO_QUBIT_GATE = 0.0008
DEPHASING_MIX_SCALE = 0.25


class PredictionResult(BaseModel):
    """Everything the system knows about one prediction run."""

    circuit_summary: dict[str, Any]
    profile_summary: dict[str, Any]
    mapping: MappingEstimate
    shots: int
    measured_qubits: list[int]
    estimated_duration_us: float
    ideal_probabilities: dict[str, float]
    predicted_probabilities: dict[str, float]
    predicted_counts: dict[str, int]
    confidence_intervals: dict[str, tuple[float, float]]
    reliability_score: float  # 0..100
    error_contributions: dict[str, float]  # per-source TVD introduced
    dominant_error_sources: list[str]
    explanations: list[str] = Field(default_factory=list)
    correction_applied: bool = False


class HardwarePredictionEngine:
    """Predicts noisy hardware output distributions for circuits."""

    def __init__(self, ideal_engine: IdealEngine | None = None):
        self.ideal = ideal_engine or IdealEngine()

    # -- timing ------------------------------------------------------------

    def estimate_duration_us(
        self, circuit: CircuitIR, profile: HardwareProfile, mapping: MappingEstimate
    ) -> float:
        t_single = profile.gate_duration_ns.get("single", 35.0)
        t_two = profile.gate_duration_ns.get("two", 300.0)
        ns = 0.0
        for layer in circuit.dependency_layers():
            has_two = any(
                circuit.instructions[i].is_two_qubit for i in layer
            )
            has_gate = any(
                not circuit.instructions[i].is_measurement for i in layer
            )
            if has_two:
                ns += t_two
            elif has_gate:
                ns += t_single
        ns += mapping.added_cnot_count * t_two
        ns += profile.measurement_duration_ns
        return ns / 1000.0

    # -- error magnitudes ----------------------------------------------------

    def _gate_error_weight(
        self, circuit: CircuitIR, profile: HardwareProfile, layout: dict[int, int]
    ) -> float:
        errors: list[float] = []
        for ins in circuit.gate_instructions:
            if ins.is_two_qubit:
                pa, pb = layout[ins.qubits[0]], layout[ins.qubits[1]]
                errors.append(profile.edge_error(pa, pb))
            else:
                p = layout[ins.qubits[0]]
                errors.append(
                    profile.single_qubit_error.get(p, profile.avg_single_qubit_error())
                )
        return 1.0 - accumulate_fidelity(errors)

    def _crosstalk_weight(
        self, circuit: CircuitIR, profile: HardwareProfile, layout: dict[int, int]
    ) -> float:
        events = 0
        for layer in circuit.dependency_layers():
            two_q = [
                circuit.instructions[i]
                for i in layer
                if circuit.instructions[i].is_two_qubit
            ]
            for i in range(len(two_q)):
                for j in range(i + 1, len(two_q)):
                    qs_i = {layout[q] for q in two_q[i].qubits}
                    qs_j = {layout[q] for q in two_q[j].qubits}
                    adjacent = any(
                        profile.has_edge(a, b) for a in qs_i for b in qs_j
                    )
                    if adjacent:
                        events += 1
        return 1.0 - (1.0 - CROSSTALK_RATE_PER_EVENT) ** events

    # -- main entry ----------------------------------------------------------

    def predict(
        self,
        circuit: CircuitIR,
        profile: HardwareProfile,
        shots: int = 1024,
        seed: int | None = None,
        correction_model: Any = None,
    ) -> PredictionResult:
        circuit.validate_structure()
        mapping = estimate_mapping(circuit, profile)
        layout = mapping.layout

        measured = sorted(circuit.measured_qubits or list(range(circuit.num_qubits)))
        m = len(measured)

        ideal_probs = self.ideal.measured_probabilities(circuit)
        vec = dist_to_vector(ideal_probs, m)
        contributions: dict[str, float] = {}
        explanations: list[str] = []

        duration_us = self.estimate_duration_us(circuit, profile, mapping)

        # 1. Gate depolarizing error (accumulated infidelity -> uniform mixing)
        before = vec.copy()
        gate_w = self._gate_error_weight(circuit, profile, layout)
        vec = mix_with_uniform(vec, gate_w)
        contributions["gate_error"] = total_variation_distance(before, vec)
        if gate_w > 0:
            explanations.append(
                f"Accumulated gate infidelity of {gate_w:.4f} across "
                f"{circuit.gate_count} gates depolarizes the output."
            )

        # 2. Topology / routing overhead from estimated SWAP insertion
        before = vec.copy()
        vec = mix_with_uniform(vec, mapping.estimated_added_gate_error)
        contributions["topology"] = total_variation_distance(before, vec)
        if mapping.added_swap_count:
            explanations.append(
                f"Routing requires ~{mapping.added_swap_count} SWAPs "
                f"({mapping.added_cnot_count} extra CNOTs), adding "
                f"{mapping.estimated_added_gate_error:.4f} error."
            )

        # 3. Crosstalk proxy: simultaneous neighboring two-qubit activity
        before = vec.copy()
        xtalk_w = self._crosstalk_weight(circuit, profile, layout)
        vec = mix_with_uniform(vec, xtalk_w)
        contributions["crosstalk"] = total_variation_distance(before, vec)
        if xtalk_w > 1e-6:
            explanations.append(
                f"Simultaneous neighboring two-qubit gates add ~{xtalk_w:.4f} "
                "crosstalk error."
            )

        # 4. Decoherence: T1 relaxation biases toward 0, T2 dephasing mixes
        before = vec.copy()
        t_s = duration_us * 1e-6
        deco_factor = 1.0
        for axis, q in enumerate(measured):
            phys = layout[q]
            t1 = profile.t1_us.get(phys, 100.0) * 1e-6
            t2 = profile.t2_us.get(phys, 80.0) * 1e-6
            p_relax = 1.0 - math.exp(-t_s / max(t1, 1e-9))
            p_deph = 1.0 - math.exp(-t_s / max(t2, 1e-9))
            vec = apply_bit_channel(vec, m, axis, relaxation_channel(p_relax))
            vec = mix_with_uniform(vec, DEPHASING_MIX_SCALE * p_deph)
            deco_factor *= (1.0 - p_relax) * (1.0 - DEPHASING_MIX_SCALE * p_deph)
        contributions["decoherence"] = total_variation_distance(before, vec)
        if contributions["decoherence"] > 1e-6:
            explanations.append(
                f"Estimated runtime {duration_us:.2f} us causes T1/T2 decay "
                f"(decoherence TVD {contributions['decoherence']:.4f})."
            )

        # 5. Leakage proxy: small mass loss redistributed uniformly
        before = vec.copy()
        leak_w = 1.0 - (1.0 - LEAKAGE_RATE_PER_TWO_QUBIT_GATE) ** (
            circuit.two_qubit_gate_count + mapping.added_cnot_count
        )
        vec = mix_with_uniform(vec, leak_w)
        contributions["leakage"] = total_variation_distance(before, vec)

        # 6. Readout error: per-qubit confusion matrices
        before = vec.copy()
        for axis, q in enumerate(measured):
            phys = layout[q]
            e = profile.readout_error.get(phys, profile.avg_readout_error())
            # Asymmetric: |1> misread slightly more often than |0> (T1 during readout)
            vec = apply_bit_channel(vec, m, axis, readout_confusion(0.8 * e, 1.2 * e))
        contributions["readout"] = total_variation_distance(before, vec)
        if contributions["readout"] > 1e-6:
            explanations.append(
                f"Per-qubit readout confusion distorts the distribution by "
                f"TVD {contributions['readout']:.4f}."
            )

        # 7. Calibration drift: stale data adds uncertainty (mixing + wider CIs)
        before = vec.copy()
        drift_u = profile.drift_uncertainty()
        vec = mix_with_uniform(vec, 0.5 * drift_u)
        contributions["drift"] = total_variation_distance(before, vec)
        if drift_u > 1e-6:
            explanations.append(
                f"Calibration is {profile.calibration_age_hours():.1f} h old; "
                f"drift adds {drift_u:.4f} fractional uncertainty."
            )

        vec = np.clip(vec, 0.0, None)
        vec /= vec.sum()

        # Optional learned correction (Workstream F)
        correction_applied = False
        if correction_model is not None:
            try:
                vec = correction_model.correct(circuit, profile, mapping, vec)
                vec = np.clip(vec, 0.0, None)
                vec /= vec.sum()
                correction_applied = True
                explanations.append(
                    "A learned correction model adjusted the predicted distribution."
                )
            except Exception as exc:  # pragma: no cover - defensive
                explanations.append(f"Correction model failed and was skipped: {exc}")

        predicted = vector_to_dist(vec, m)

        # Reliability score: survival probability across all modeled channels
        survival = (
            (1.0 - gate_w)
            * (1.0 - mapping.estimated_added_gate_error)
            * (1.0 - xtalk_w)
            * deco_factor
            * (1.0 - leak_w)
            * accumulate_fidelity(
                [
                    profile.readout_error.get(layout[q], profile.avg_readout_error())
                    for q in measured
                ]
            )
            * (1.0 - 0.5 * drift_u)
        )
        reliability = round(100.0 * max(0.0, min(1.0, survival)), 2)

        dominant = sorted(contributions, key=lambda k: -contributions[k])
        dominant = [k for k in dominant if contributions[k] > 1e-6][:3]

        cis = confidence_intervals(predicted, shots, widen_factor=1.0 + drift_u)
        counts = sample_counts(predicted, shots, seed=seed)

        return PredictionResult(
            circuit_summary=circuit.summary(),
            profile_summary=profile.summary(),
            mapping=mapping,
            shots=shots,
            measured_qubits=measured,
            estimated_duration_us=round(duration_us, 4),
            ideal_probabilities=ideal_probs,
            predicted_probabilities=predicted,
            predicted_counts=counts,
            confidence_intervals=cis,
            reliability_score=reliability,
            error_contributions={k: round(v, 6) for k, v in contributions.items()},
            dominant_error_sources=dominant,
            explanations=explanations,
            correction_applied=correction_applied,
        )

"""Deterministic analysis agents and the council that combines them.

Each agent inspects the circuit, hardware profile, and prediction result
from one perspective and returns a structured report. The CouncilEngine
merges the reports into a hardware execution assessment.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.circuits.ir import CircuitIR
from app.hardware.profile import HardwareProfile
from app.prediction.engine import PredictionResult


class AgentReport(BaseModel):
    agent: str
    summary: str
    evidence: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    confidence: float  # 0..1
    mitigation: str = ""


class CouncilReport(BaseModel):
    reliability_classification: str  # high / moderate / low
    reliability_score: float
    agent_reports: list[AgentReport]
    mitigation_strategy: list[str]
    beginner_explanation: str
    advanced_explanation: str


class BaseAgent:
    name = "BaseAgent"

    def analyze(
        self, circuit: CircuitIR, profile: HardwareProfile, result: PredictionResult
    ) -> AgentReport:
        raise NotImplementedError


class HardwareNoiseAgent(BaseAgent):
    name = "HardwareNoiseAgent"

    def analyze(self, circuit, profile, result) -> AgentReport:
        gate_tvd = result.error_contributions.get("gate_error", 0.0)
        avg_2q = profile.avg_two_qubit_error()
        evidence = [
            f"{circuit.gate_count} gates, {circuit.two_qubit_gate_count} two-qubit",
            f"average two-qubit error {avg_2q:.4f}",
            f"gate-error contribution (TVD) {gate_tvd:.4f}",
        ]
        concerns = []
        if circuit.two_qubit_gate_count * avg_2q > 0.05:
            concerns.append("accumulated two-qubit infidelity exceeds 5%")
        return AgentReport(
            agent=self.name,
            summary=(
                "Gate noise is the dominant channel"
                if "gate_error" in result.dominant_error_sources[:1]
                else "Gate noise is present but not dominant"
            ),
            evidence=evidence,
            concerns=concerns,
            confidence=0.85 if circuit.gate_count < 50 else 0.7,
            mitigation="Reduce two-qubit gate count; merge adjacent rotations.",
        )


class TopologyAgent(BaseAgent):
    name = "TopologyAgent"

    def analyze(self, circuit, profile, result) -> AgentReport:
        m = result.mapping
        evidence = [
            f"layout {m.layout}",
            f"{m.nonnative_two_qubit_gates} non-native two-qubit gate(s)",
            f"~{m.added_swap_count} SWAP(s) / {m.added_cnot_count} extra CNOT(s)",
            f"topology penalty {m.topology_penalty}",
        ]
        concerns = list(m.warnings)
        return AgentReport(
            agent=self.name,
            summary=(
                "Circuit fits the topology natively"
                if m.hardware_friendly
                else "Circuit requires significant routing overhead"
            ),
            evidence=evidence,
            concerns=concerns,
            confidence=0.9,
            mitigation=(
                "No routing changes needed."
                if m.added_swap_count == 0
                else "Reorder logical qubits to match the coupling map or pick "
                "a better-connected backend."
            ),
        )


class ReadoutAgent(BaseAgent):
    name = "ReadoutAgent"

    def analyze(self, circuit, profile, result) -> AgentReport:
        ro_tvd = result.error_contributions.get("readout", 0.0)
        worst_q = max(profile.readout_error, key=profile.readout_error.get)
        evidence = [
            f"average readout error {profile.avg_readout_error():.4f}",
            f"worst readout qubit q{worst_q} at {profile.readout_error[worst_q]:.4f}",
            f"readout contribution (TVD) {ro_tvd:.4f}",
        ]
        concerns = []
        if profile.avg_readout_error() > 0.03:
            concerns.append("readout error above 3% will visibly distort histograms")
        return AgentReport(
            agent=self.name,
            summary=(
                "Readout error is the leading distortion"
                if result.dominant_error_sources[:1] == ["readout"]
                else "Readout error contributes moderately"
            ),
            evidence=evidence,
            concerns=concerns,
            confidence=0.88,
            mitigation="Apply measurement-error mitigation (confusion-matrix inversion).",
        )


class DecoherenceAgent(BaseAgent):
    name = "DecoherenceAgent"

    def analyze(self, circuit, profile, result) -> AgentReport:
        deco_tvd = result.error_contributions.get("decoherence", 0.0)
        avg_t1 = sum(profile.t1_us.values()) / len(profile.t1_us)
        evidence = [
            f"estimated runtime {result.estimated_duration_us:.2f} us",
            f"average T1 {avg_t1:.1f} us",
            f"decoherence contribution (TVD) {deco_tvd:.4f}",
        ]
        concerns = []
        if result.estimated_duration_us > 0.05 * avg_t1:
            concerns.append("circuit runtime exceeds 5% of T1; relaxation bias toward |0>")
        return AgentReport(
            agent=self.name,
            summary=(
                "Decoherence materially biases outputs toward the ground state"
                if deco_tvd > 0.02
                else "Decoherence impact is small for this circuit length"
            ),
            evidence=evidence,
            concerns=concerns,
            confidence=0.8,
            mitigation="Shorten the critical path; schedule gates densely to cut idle time.",
        )


class DriftAgent(BaseAgent):
    name = "DriftAgent"

    def analyze(self, circuit, profile, result) -> AgentReport:
        age = profile.calibration_age_hours()
        drift = profile.drift_uncertainty()
        evidence = [
            f"calibration age {age:.1f} h",
            f"drift uncertainty {drift:.4f}",
        ]
        concerns = []
        if age > 24:
            concerns.append("calibration data older than 24 h; expect parameter drift")
        return AgentReport(
            agent=self.name,
            summary=(
                "Calibration is fresh; drift negligible"
                if drift < 0.01
                else "Stale calibration widens prediction uncertainty"
            ),
            evidence=evidence,
            concerns=concerns,
            confidence=max(0.5, 0.95 - drift * 2),
            mitigation="Re-run device calibration or widen confidence intervals.",
        )


class LearningCorrectionAgent(BaseAgent):
    name = "LearningCorrectionAgent"

    def analyze(self, circuit, profile, result) -> AgentReport:
        applied = result.correction_applied
        return AgentReport(
            agent=self.name,
            summary=(
                "A trained correction model refined this prediction"
                if applied
                else "No learned correction was applied (rule-based prediction only)"
            ),
            evidence=[f"correction_applied={applied}"],
            concerns=(
                []
                if applied
                else ["prediction relies solely on analytic noise models"]
            ),
            confidence=0.75 if applied else 0.6,
            mitigation=(
                "Keep retraining on fresh synthetic or real calibration data."
                if applied
                else "Train the correction layer on synthetic runs (POST /train/synthetic)."
            ),
        )


class TeachingAgent(BaseAgent):
    name = "TeachingAgent"

    def analyze(self, circuit, profile, result) -> AgentReport:
        top = result.dominant_error_sources[0] if result.dominant_error_sources else "none"
        friendly = {
            "gate_error": "imperfect gates slightly scramble the answer",
            "readout": "the measurement itself sometimes reads a 0 as 1 or vice versa",
            "decoherence": "qubits forget their state over time, drifting toward 0",
            "topology": "extra SWAP operations were needed to connect distant qubits",
            "crosstalk": "neighboring operations interfere with each other",
            "leakage": "a tiny amount of probability escapes the qubit levels",
            "drift": "the device has changed since it was last calibrated",
            "none": "this circuit is short enough that noise barely matters",
        }
        return AgentReport(
            agent=self.name,
            summary=f"Biggest effect in plain words: {friendly.get(top, top)}.",
            evidence=[
                f"reliability score {result.reliability_score}/100",
                f"dominant sources: {', '.join(result.dominant_error_sources) or 'none'}",
            ],
            concerns=[],
            confidence=0.9,
            mitigation="Compare the ideal vs predicted histograms to see the distortion.",
        )


ALL_AGENTS: list[BaseAgent] = [
    HardwareNoiseAgent(),
    TopologyAgent(),
    ReadoutAgent(),
    DecoherenceAgent(),
    DriftAgent(),
    LearningCorrectionAgent(),
    TeachingAgent(),
]


class CouncilEngine:
    """Runs every agent and merges their reports."""

    def __init__(self, agents: list[BaseAgent] | None = None):
        self.agents = agents or ALL_AGENTS

    def convene(
        self, circuit: CircuitIR, profile: HardwareProfile, result: PredictionResult
    ) -> CouncilReport:
        reports = [a.analyze(circuit, profile, result) for a in self.agents]

        score = result.reliability_score
        if score >= 85:
            classification = "high"
        elif score >= 60:
            classification = "moderate"
        else:
            classification = "low"

        mitigations = []
        for r in reports:
            if r.concerns and r.mitigation and r.mitigation not in mitigations:
                mitigations.append(r.mitigation)
        if not mitigations:
            mitigations.append("No mitigation required; expected fidelity is high.")

        top = result.dominant_error_sources[0] if result.dominant_error_sources else "none"
        beginner = (
            f"If you ran this circuit on '{profile.backend_name}', about "
            f"{score:.0f}% of shots would behave as intended. The main source "
            f"of disturbance is {top.replace('_', ' ')}. The predicted histogram "
            "shows what the device would actually return, not the textbook answer."
        )
        advanced = (
            f"Survival-probability reliability {score:.2f}/100 on "
            f"{profile.backend_name}. Channel TVD contributions: "
            + ", ".join(
                f"{k}={v:.4f}" for k, v in result.error_contributions.items()
            )
            + f". Mapping added {result.mapping.added_swap_count} SWAPs "
            f"(topology penalty {result.mapping.topology_penalty}). "
            f"CIs widened by drift factor {1 + profile.drift_uncertainty():.3f}."
        )

        return CouncilReport(
            reliability_classification=classification,
            reliability_score=score,
            agent_reports=reports,
            mitigation_strategy=mitigations,
            beginner_explanation=beginner,
            advanced_explanation=advanced,
        )


def council_to_dict(report: CouncilReport) -> dict[str, Any]:
    return report.model_dump()

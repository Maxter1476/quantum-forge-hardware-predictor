"""Markdown report generation for prediction runs."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.agents.council import CouncilReport
from app.circuits.ir import CircuitIR
from app.hardware.profile import HardwareProfile
from app.prediction.engine import PredictionResult

REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "reports"


def _dist_table(
    ideal: dict[str, float],
    predicted: dict[str, float],
    counts: dict[str, int],
    cis: dict[str, tuple[float, float]],
) -> str:
    keys = sorted(set(ideal) | set(predicted))
    lines = [
        "| bitstring | ideal P | predicted P | predicted counts | 95% CI |",
        "|-----------|---------|-------------|------------------|--------|",
    ]
    for k in keys:
        lo, hi = cis.get(k, (0.0, 0.0))
        lines.append(
            f"| `{k}` | {ideal.get(k, 0.0):.4f} | {predicted.get(k, 0.0):.4f} "
            f"| {counts.get(k, 0)} | [{lo:.4f}, {hi:.4f}] |"
        )
    return "\n".join(lines)


def generate_prediction_report(
    circuit: CircuitIR,
    profile: HardwareProfile,
    result: PredictionResult,
    council: CouncilReport,
) -> str:
    """Render one prediction run as a Markdown document."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cs = result.circuit_summary
    ps = result.profile_summary
    m = result.mapping

    sections = [
        f"# Hardware Prediction Report: `{circuit.name}` on `{profile.backend_name}`",
        f"_Generated {now} by Quantum Forge Hardware Predictor_",
        "## Circuit Summary",
        f"- qubits: {cs['num_qubits']}  \n- depth: {cs['depth']}  \n"
        f"- gates: {cs['gate_count']} ({cs['two_qubit_gate_count']} two-qubit)  \n"
        f"- measured qubits: {result.measured_qubits}  \n- shots: {result.shots}",
        "## Hardware Profile",
        "\n".join(f"- {k}: {v}" for k, v in ps.items()),
        "## Topology Fit",
        f"- layout: {m.layout}  \n- native 2q gates: {m.native_two_qubit_gates}  \n"
        f"- non-native 2q gates: {m.nonnative_two_qubit_gates}  \n"
        f"- est. SWAPs added: {m.added_swap_count} ({m.added_cnot_count} CNOTs)  \n"
        f"- est. depth after mapping: {m.estimated_depth_after_mapping}  \n"
        f"- topology penalty: {m.topology_penalty}  \n"
        f"- hardware friendly: {m.hardware_friendly}",
        "## Output Distributions",
        _dist_table(
            result.ideal_probabilities,
            result.predicted_probabilities,
            result.predicted_counts,
            result.confidence_intervals,
        ),
        "## Error Source Breakdown (TVD contribution)",
        "\n".join(
            f"- **{k}**: {v:.4f}" for k, v in result.error_contributions.items()
        ),
        f"\n**Dominant sources:** {', '.join(result.dominant_error_sources) or 'none'}",
        f"\n**Reliability score:** {result.reliability_score}/100 "
        f"({council.reliability_classification})",
        "## Engine Explanations",
        "\n".join(f"- {e}" for e in result.explanations) or "- (none)",
        "## Agent Council Analysis",
    ]
    for r in council.agent_reports:
        block = [f"### {r.agent}", f"{r.summary} _(confidence {r.confidence:.2f})_"]
        if r.evidence:
            block.append("Evidence:\n" + "\n".join(f"- {e}" for e in r.evidence))
        if r.concerns:
            block.append("Concerns:\n" + "\n".join(f"- {c}" for c in r.concerns))
        if r.mitigation:
            block.append(f"Mitigation: {r.mitigation}")
        sections.append("\n\n".join(block))

    sections += [
        "## Mitigation Strategy",
        "\n".join(f"1. {s}" for s in council.mitigation_strategy),
        "## Plain-Language Summary",
        council.beginner_explanation,
        "## Technical Summary",
        council.advanced_explanation,
    ]
    return "\n\n".join(sections) + "\n"


def save_report(markdown: str, report_id: str, directory: Path | None = None) -> Path:
    directory = directory or REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report_id}.md"
    path.write_text(markdown)
    return path

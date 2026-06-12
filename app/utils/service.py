"""High-level orchestration shared by the API and the dashboard.

One call runs the whole pipeline: parse -> map -> predict -> council ->
report -> persist.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agents.council import CouncilEngine, CouncilReport
from app.circuits.dsl import parse_dsl
from app.circuits.ir import CircuitIR
from app.hardware.registry import get_profile
from app.hardware.profile import HardwareProfile
from app.learning.correction import CorrectionModel
from app.learning.synthetic import generate_dataset
from app.prediction.engine import HardwarePredictionEngine, PredictionResult
from app.prediction.mitigation import mitigate_readout
from app.reports.markdown import generate_prediction_report, save_report
from app.storage import db as storage

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def default_model_path(backend_name: str, model_type: str) -> Path:
    return MODELS_DIR / f"correction_{backend_name}_{model_type}.pkl"


def load_correction_model(backend_name: str) -> CorrectionModel | None:
    """Load the most recently trained correction model for a backend, if any."""
    if not MODELS_DIR.exists():
        return None
    candidates = sorted(
        MODELS_DIR.glob(f"correction_{backend_name}_*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    try:
        return CorrectionModel.load(candidates[0])
    except Exception:
        return None


def resolve_circuit(
    dsl: str | None = None, ir: dict[str, Any] | None = None
) -> CircuitIR:
    if dsl:
        return parse_dsl(dsl)
    if ir:
        return CircuitIR.from_json(ir)
    raise ValueError("provide either 'dsl' source or 'ir' JSON")


def run_prediction(
    circuit: CircuitIR,
    profile: HardwareProfile,
    shots: int = 1024,
    seed: int | None = None,
    use_correction: bool = False,
    dsl_source: str = "",
    persist: bool = True,
    generate_report: bool = False,
    apply_readout_mitigation: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline; returns result, council, and optional report."""
    engine = HardwarePredictionEngine()
    correction = load_correction_model(profile.backend_name) if use_correction else None
    result = engine.predict(
        circuit, profile, shots=shots, seed=seed, correction_model=correction
    )
    council = CouncilEngine().convene(circuit, profile, result)
    mitigation = (
        mitigate_readout(result, profile) if apply_readout_mitigation else None
    )

    run_id = None
    report_id = None
    markdown = None
    if generate_report:
        markdown = generate_prediction_report(circuit, profile, result, council)

    if persist:
        run_id, report_id = _persist_run(
            circuit, profile, result, council, dsl_source, markdown
        )

    payload: dict[str, Any] = {
        "run_id": run_id,
        "result": result.model_dump(),
        "council": council.model_dump(),
    }
    if mitigation is not None:
        payload["readout_mitigation"] = mitigation
    if markdown is not None:
        payload["report_id"] = report_id
        payload["report_markdown"] = markdown
    return payload


def compare_backends(
    circuit: CircuitIR,
    backend_names: list[str],
    shots: int = 1024,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Predict the same circuit on several backends for side-by-side review.

    Backends that cannot host the circuit (too few qubits) are reported with
    an error message instead of failing the whole comparison.
    """
    engine = HardwarePredictionEngine()
    rows: list[dict[str, Any]] = []
    for name in backend_names:
        profile = get_profile(name)
        try:
            result = engine.predict(circuit, profile, shots=shots, seed=seed)
        except ValueError as exc:
            rows.append({"backend": name, "error": str(exc)})
            continue
        rows.append(
            {
                "backend": name,
                "reliability_score": result.reliability_score,
                "predicted_probabilities": result.predicted_probabilities,
                "dominant_error_sources": result.dominant_error_sources,
                "added_swap_count": result.mapping.added_swap_count,
                "estimated_duration_us": result.estimated_duration_us,
            }
        )
    rows.sort(key=lambda r: -(r.get("reliability_score") or -1))
    return rows


def _persist_run(
    circuit: CircuitIR,
    profile: HardwareProfile,
    result: PredictionResult,
    council: CouncilReport,
    dsl_source: str,
    markdown: str | None,
) -> tuple[str, str | None]:
    session = storage.get_session()
    try:
        circuit_row = storage.CircuitRow(
            name=circuit.name,
            dsl_source=dsl_source,
            ir_json=storage.to_jsonable(circuit.model_dump()),
        )
        session.add(circuit_row)
        session.flush()

        run = storage.PredictionRunRow(
            circuit_id=circuit_row.id,
            backend_name=profile.backend_name,
            shots=result.shots,
            reliability_score=result.reliability_score,
            result_json=storage.to_jsonable(result.model_dump()),
        )
        session.add(run)
        session.flush()

        for bits in sorted(
            set(result.ideal_probabilities) | set(result.predicted_probabilities)
        ):
            lo, hi = result.confidence_intervals.get(bits, (0.0, 0.0))
            session.add(
                storage.PredictedOutputRow(
                    run_id=run.id,
                    bitstring=bits,
                    ideal_probability=result.ideal_probabilities.get(bits, 0.0),
                    predicted_probability=result.predicted_probabilities.get(bits, 0.0),
                    predicted_count=result.predicted_counts.get(bits, 0),
                    ci_low=lo,
                    ci_high=hi,
                )
            )
        for r in council.agent_reports:
            session.add(
                storage.AgentReportRow(
                    run_id=run.id,
                    agent=r.agent,
                    report_json=storage.to_jsonable(r.model_dump()),
                )
            )

        report_id = None
        if markdown is not None:
            report_row = storage.GeneratedReportRow(
                run_id=run.id,
                title=f"{circuit.name} on {profile.backend_name}",
                markdown=markdown,
            )
            session.add(report_row)
            session.flush()
            path = save_report(markdown, report_row.id)
            report_row.file_path = str(path)
            report_id = report_row.id

        session.commit()
        return run.id, report_id
    finally:
        session.close()


def train_synthetic(
    backend_name: str,
    num_examples: int = 60,
    shots: int = 1024,
    model_type: str = "random_forest",
    seed: int = 0,
) -> dict[str, Any]:
    """Generate synthetic data, train a correction model, persist both."""
    profile = get_profile(backend_name)
    examples = generate_dataset(
        profile, num_examples=num_examples, shots=shots, seed=seed
    )
    model = CorrectionModel(model_type=model_type)
    metrics = model.train(examples)
    path = default_model_path(backend_name, model_type)
    model.save(path)

    session = storage.get_session()
    try:
        metrics_json = storage.to_jsonable(vars(metrics))
        session.add(
            storage.SyntheticTrainingRunRow(
                backend_name=backend_name,
                num_examples=num_examples,
                model_type=model_type,
                metrics_json=metrics_json,
            )
        )
        session.add(
            storage.ModelRegistryRow(
                model_type=model_type,
                backend_name=backend_name,
                file_path=str(path),
                metrics_json=metrics_json,
            )
        )
        session.commit()
    finally:
        session.close()

    return {
        "backend_name": backend_name,
        "model_type": model_type,
        "model_path": str(path),
        "metrics": vars(metrics),
    }


def list_reports() -> list[dict[str, Any]]:
    session = storage.get_session()
    try:
        rows = (
            session.query(storage.GeneratedReportRow)
            .order_by(storage.GeneratedReportRow.created_at.desc())
            .all()
        )
        return [
            {
                "report_id": r.id,
                "run_id": r.run_id,
                "title": r.title,
                "created_at": str(r.created_at),
            }
            for r in rows
        ]
    finally:
        session.close()


def get_report(report_id: str) -> dict[str, Any] | None:
    session = storage.get_session()
    try:
        row = session.get(storage.GeneratedReportRow, report_id)
        if row is None:
            return None
        return {
            "report_id": row.id,
            "run_id": row.run_id,
            "title": row.title,
            "markdown": row.markdown,
            "created_at": str(row.created_at),
        }
    finally:
        session.close()

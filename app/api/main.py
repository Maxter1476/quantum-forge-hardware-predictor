"""FastAPI backend for the Quantum Forge Hardware Predictor."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.circuits.dsl import DSLParseError
from app.hardware.registry import get_profile, list_profiles, register_profile
from app.storage import db as storage
from app.utils import service

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

@asynccontextmanager
async def _lifespan(_: FastAPI):
    storage.init_db()
    yield


app = FastAPI(
    title="Quantum Forge Hardware Predictor",
    description=(
        "Predicts what a real quantum processor would output for a circuit, "
        "using hardware-inspired noise models, topology constraints, and "
        "an optional learned correction layer."
    ),
    version="0.1.0",
    lifespan=_lifespan,
)


class ParseRequest(BaseModel):
    dsl: str


class PredictRequest(BaseModel):
    dsl: str | None = None
    ir: dict[str, Any] | None = None
    backend: str = "toy-5q"
    shots: int = Field(default=1024, ge=1, le=1_000_000)
    seed: int | None = None
    use_correction: bool = False
    apply_readout_mitigation: bool = False


class CompareRequest(BaseModel):
    dsl: str | None = None
    ir: dict[str, Any] | None = None
    backends: list[str] = Field(min_length=1, max_length=10)
    shots: int = Field(default=1024, ge=1, le=1_000_000)
    seed: int | None = None


class RegisterProfileRequest(BaseModel):
    profile: dict[str, Any]
    overwrite: bool = False


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    backend: str = "toy-5q"
    num_examples: int = Field(default=60, ge=4, le=2000)
    shots: int = Field(default=1024, ge=16, le=100_000)
    model_type: str = "random_forest"
    seed: int = 0


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "quantum-forge-hardware-predictor"}


@app.get("/hardware-profiles")
def hardware_profiles() -> dict[str, Any]:
    return {
        "profiles": [get_profile(name).summary() for name in list_profiles()]
    }


@app.get("/hardware-profiles/{name}")
def hardware_profile(name: str) -> dict[str, Any]:
    try:
        return get_profile(name).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/hardware-profiles", status_code=201)
def create_hardware_profile(req: RegisterProfileRequest) -> dict[str, Any]:
    """Register a custom backend, e.g. transcribed real calibration data."""
    try:
        profile = register_profile(req.profile, overwrite=req.overwrite)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"registered": profile.backend_name, "summary": profile.summary()}


@app.post("/circuits/parse")
def parse_circuit(req: ParseRequest) -> dict[str, Any]:
    try:
        circuit = service.resolve_circuit(dsl=req.dsl)
    except (DSLParseError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ir": circuit.model_dump(), "summary": circuit.summary()}


def _run(req: PredictRequest, generate_report: bool) -> dict[str, Any]:
    try:
        circuit = service.resolve_circuit(dsl=req.dsl, ir=req.ir)
        profile = get_profile(req.backend)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (DSLParseError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        return service.run_prediction(
            circuit,
            profile,
            shots=req.shots,
            seed=req.seed,
            use_correction=req.use_correction,
            dsl_source=req.dsl or "",
            generate_report=generate_report,
            apply_readout_mitigation=req.apply_readout_mitigation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    return _run(req, generate_report=False)


@app.post("/predict/report")
def predict_with_report(req: PredictRequest) -> dict[str, Any]:
    return _run(req, generate_report=True)


@app.post("/predict/compare")
def predict_compare(req: CompareRequest) -> dict[str, Any]:
    """Predict one circuit on several backends, sorted by reliability."""
    try:
        circuit = service.resolve_circuit(dsl=req.dsl, ir=req.ir)
        rows = service.compare_backends(
            circuit, req.backends, shots=req.shots, seed=req.seed
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (DSLParseError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"circuit": circuit.summary(), "comparison": rows}


@app.post("/train/synthetic")
def train_synthetic(req: TrainRequest) -> dict[str, Any]:
    try:
        return service.train_synthetic(
            backend_name=req.backend,
            num_examples=req.num_examples,
            shots=req.shots,
            model_type=req.model_type,
            seed=req.seed,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/reports")
def reports() -> dict[str, Any]:
    return {"reports": service.list_reports()}


@app.get("/reports/{report_id}")
def report(report_id: str) -> dict[str, Any]:
    found = service.get_report(report_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"no report '{report_id}'")
    return found


@app.get("/examples")
def examples() -> dict[str, Any]:
    out = {}
    if EXAMPLES_DIR.exists():
        for path in sorted(EXAMPLES_DIR.glob("*.qf")):
            out[path.stem] = path.read_text()
    return {"examples": out}

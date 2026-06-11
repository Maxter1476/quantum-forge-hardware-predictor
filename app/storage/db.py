"""SQLAlchemy models and session helpers (SQLite)."""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_DB_URL = f"sqlite:///{DATA_DIR / 'qfhp.db'}"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Base(DeclarativeBase):
    pass


class CircuitRow(Base):
    __tablename__ = "circuits"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(String)
    dsl_source: Mapped[str] = mapped_column(Text, default="")
    ir_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class HardwareProfileRow(Base):
    __tablename__ = "hardware_profiles"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    backend_name: Mapped[str] = mapped_column(String, unique=True)
    profile_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class PredictionRunRow(Base):
    __tablename__ = "prediction_runs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    circuit_id: Mapped[str | None] = mapped_column(ForeignKey("circuits.id"), nullable=True)
    backend_name: Mapped[str] = mapped_column(String)
    shots: Mapped[int] = mapped_column(Integer)
    reliability_score: Mapped[float] = mapped_column(Float)
    result_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class PredictedOutputRow(Base):
    __tablename__ = "predicted_outputs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("prediction_runs.id"))
    bitstring: Mapped[str] = mapped_column(String)
    ideal_probability: Mapped[float] = mapped_column(Float)
    predicted_probability: Mapped[float] = mapped_column(Float)
    predicted_count: Mapped[int] = mapped_column(Integer)
    ci_low: Mapped[float] = mapped_column(Float)
    ci_high: Mapped[float] = mapped_column(Float)


class AgentReportRow(Base):
    __tablename__ = "agent_reports"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("prediction_runs.id"))
    agent: Mapped[str] = mapped_column(String)
    report_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class GeneratedReportRow(Base):
    __tablename__ = "generated_reports"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("prediction_runs.id"), nullable=True
    )
    title: Mapped[str] = mapped_column(String)
    markdown: Mapped[str] = mapped_column(Text)
    file_path: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class SyntheticTrainingRunRow(Base):
    __tablename__ = "synthetic_training_runs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    backend_name: Mapped[str] = mapped_column(String)
    num_examples: Mapped[int] = mapped_column(Integer)
    model_type: Mapped[str] = mapped_column(String)
    metrics_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ModelRegistryRow(Base):
    __tablename__ = "model_registry"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    model_type: Mapped[str] = mapped_column(String)
    backend_name: Mapped[str] = mapped_column(String)
    file_path: Mapped[str] = mapped_column(String)
    metrics_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class DevelopmentEventRow(Base):
    __tablename__ = "development_events"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    event: Mapped[str] = mapped_column(Text)
    details_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


_engine = None
_SessionLocal: sessionmaker | None = None


def init_db(db_url: str | None = None):
    """Create the engine, tables, and session factory. Idempotent."""
    global _engine, _SessionLocal
    db_url = db_url or os.environ.get("QFHP_DB_URL", DEFAULT_DB_URL)
    if db_url.startswith("sqlite:///") and ":memory:" not in db_url:
        Path(db_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_session() -> Session:
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


def to_jsonable(obj: Any) -> Any:
    """Round-trip through JSON to coerce datetimes/tuples for the JSON column."""
    return json.loads(json.dumps(obj, default=str))

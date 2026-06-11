"""Hardware backend profile model.

A profile captures everything the prediction engine needs to know about a
(mock) quantum processor: topology, error rates, coherence times, timing,
and how stale its calibration data is.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DriftModel(BaseModel):
    """How prediction uncertainty grows as calibration data ages."""

    error_growth_per_hour: float = 0.0005
    readout_growth_per_hour: float = 0.0003
    max_extra_uncertainty: float = 0.15


class HardwareProfile(BaseModel):
    """Full description of a mock quantum backend."""

    backend_name: str
    num_qubits: int
    coupling_map: list[tuple[int, int]]
    basis_gates: list[str] = Field(
        default_factory=lambda: ["rz", "sx", "x", "cx"]
    )
    single_qubit_error: dict[int, float]
    two_qubit_error: dict[str, float]  # key "a-b" with a < b
    readout_error: dict[int, float]
    t1_us: dict[int, float]
    t2_us: dict[int, float]
    gate_duration_ns: dict[str, float] = Field(
        default_factory=lambda: {"single": 35.0, "two": 300.0}
    )
    measurement_duration_ns: float = 700.0
    calibration_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    drift_model: DriftModel = Field(default_factory=DriftModel)
    description: str = ""

    @field_validator("coupling_map")
    @classmethod
    def _normalize_edges(cls, v: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(min(a, b), max(a, b)) for a, b in v]

    # -- lookups -----------------------------------------------------------

    @staticmethod
    def edge_key(a: int, b: int) -> str:
        return f"{min(a, b)}-{max(a, b)}"

    def has_edge(self, a: int, b: int) -> bool:
        return (min(a, b), max(a, b)) in set(self.coupling_map)

    def neighbors(self, q: int) -> list[int]:
        out = []
        for a, b in self.coupling_map:
            if a == q:
                out.append(b)
            elif b == q:
                out.append(a)
        return sorted(set(out))

    def edge_error(self, a: int, b: int) -> float:
        key = self.edge_key(a, b)
        if key in self.two_qubit_error:
            return self.two_qubit_error[key]
        return self.avg_two_qubit_error()

    def avg_single_qubit_error(self) -> float:
        vals = list(self.single_qubit_error.values())
        return sum(vals) / len(vals) if vals else 0.0

    def avg_two_qubit_error(self) -> float:
        vals = list(self.two_qubit_error.values())
        return sum(vals) / len(vals) if vals else 0.0

    def avg_readout_error(self) -> float:
        vals = list(self.readout_error.values())
        return sum(vals) / len(vals) if vals else 0.0

    def calibration_age_hours(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        ts = self.calibration_timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (now - ts).total_seconds() / 3600.0)

    def drift_uncertainty(self, now: datetime | None = None) -> float:
        """Extra fractional uncertainty caused by stale calibration."""
        age = self.calibration_age_hours(now)
        extra = age * (
            self.drift_model.error_growth_per_hour
            + self.drift_model.readout_growth_per_hour
        )
        return min(extra, self.drift_model.max_extra_uncertainty)

    def summary(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "num_qubits": self.num_qubits,
            "edges": len(self.coupling_map),
            "basis_gates": self.basis_gates,
            "avg_single_qubit_error": round(self.avg_single_qubit_error(), 6),
            "avg_two_qubit_error": round(self.avg_two_qubit_error(), 6),
            "avg_readout_error": round(self.avg_readout_error(), 6),
            "avg_t1_us": round(sum(self.t1_us.values()) / len(self.t1_us), 2),
            "avg_t2_us": round(sum(self.t2_us.values()) / len(self.t2_us), 2),
            "calibration_age_hours": round(self.calibration_age_hours(), 2),
            "description": self.description,
        }

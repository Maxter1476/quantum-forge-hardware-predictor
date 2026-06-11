"""Circuit intermediate representation (IR).

The IR is the lingua franca of the system: the DSL parser produces it, the
transpilation estimator analyses it, the ideal engine simulates it, and the
noise engine predicts hardware output from it.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

SINGLE_QUBIT_GATES = {"i", "x", "y", "z", "h", "s", "t", "rx", "ry", "rz"}
TWO_QUBIT_GATES = {"cx", "cnot", "cz", "swap"}
PARAMETRIC_GATES = {"rx", "ry", "rz"}
SUPPORTED_GATES = SINGLE_QUBIT_GATES | TWO_QUBIT_GATES | {"measure"}


class Instruction(BaseModel):
    """A single gate or measurement instruction."""

    name: str
    qubits: list[int]
    clbits: list[int] = Field(default_factory=list)
    params: list[float] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, v: str) -> str:
        v = v.lower()
        if v == "cnot":
            v = "cx"
        if v not in SUPPORTED_GATES and v != "cx":
            raise ValueError(f"unsupported gate: {v}")
        return v

    @property
    def is_measurement(self) -> bool:
        return self.name == "measure"

    @property
    def is_two_qubit(self) -> bool:
        return self.name in {"cx", "cz", "swap"}


class CircuitIR(BaseModel):
    """A quantum circuit as an ordered list of instructions."""

    name: str = "circuit"
    num_qubits: int
    num_clbits: int = 0
    instructions: list[Instruction] = Field(default_factory=list)

    @field_validator("num_qubits")
    @classmethod
    def _positive_qubits(cls, v: int) -> int:
        if v < 1:
            raise ValueError("circuit needs at least one qubit")
        return v

    def validate_structure(self) -> None:
        """Raise ValueError if any instruction references invalid wires."""
        for idx, ins in enumerate(self.instructions):
            for q in ins.qubits:
                if not 0 <= q < self.num_qubits:
                    raise ValueError(
                        f"instruction {idx} ({ins.name}) references qubit {q} "
                        f"but circuit has {self.num_qubits} qubits"
                    )
            for c in ins.clbits:
                if not 0 <= c < self.num_clbits:
                    raise ValueError(
                        f"instruction {idx} ({ins.name}) references clbit {c} "
                        f"but circuit has {self.num_clbits} clbits"
                    )
            if ins.name in PARAMETRIC_GATES and len(ins.params) != 1:
                raise ValueError(f"gate {ins.name} requires exactly one parameter")
            if ins.is_two_qubit and len(ins.qubits) != 2:
                raise ValueError(f"gate {ins.name} requires exactly two qubits")
            if ins.is_two_qubit and ins.qubits[0] == ins.qubits[1]:
                raise ValueError(f"gate {ins.name} acts twice on qubit {ins.qubits[0]}")
            if ins.name in SINGLE_QUBIT_GATES and len(ins.qubits) != 1:
                raise ValueError(f"gate {ins.name} requires exactly one qubit")

    # -- analytics ---------------------------------------------------------

    @property
    def gate_instructions(self) -> list[Instruction]:
        return [i for i in self.instructions if not i.is_measurement]

    @property
    def gate_count(self) -> int:
        return len(self.gate_instructions)

    @property
    def two_qubit_gate_count(self) -> int:
        return sum(1 for i in self.gate_instructions if i.is_two_qubit)

    @property
    def measured_qubits(self) -> list[int]:
        seen: list[int] = []
        for ins in self.instructions:
            if ins.is_measurement:
                for q in ins.qubits:
                    if q not in seen:
                        seen.append(q)
        return seen

    def qubit_usage(self) -> dict[int, int]:
        """Number of gate instructions touching each qubit."""
        usage = {q: 0 for q in range(self.num_qubits)}
        for ins in self.gate_instructions:
            for q in ins.qubits:
                usage[q] += 1
        return usage

    def dependency_layers(self) -> list[list[int]]:
        """Greedy ASAP layering of instruction indices (defines depth)."""
        layers: list[list[int]] = []
        qubit_layer = [0] * self.num_qubits
        for idx, ins in enumerate(self.instructions):
            start = max((qubit_layer[q] for q in ins.qubits), default=0)
            while len(layers) <= start:
                layers.append([])
            layers[start].append(idx)
            for q in ins.qubits:
                qubit_layer[q] = start + 1
        return layers

    @property
    def depth(self) -> int:
        return len(self.dependency_layers())

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "num_clbits": self.num_clbits,
            "depth": self.depth,
            "gate_count": self.gate_count,
            "two_qubit_gate_count": self.two_qubit_gate_count,
            "qubit_usage": self.qubit_usage(),
            "measured_qubits": self.measured_qubits,
        }

    # -- serialization -----------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=2)

    @classmethod
    def from_json(cls, payload: str | dict[str, Any]) -> "CircuitIR":
        data = json.loads(payload) if isinstance(payload, str) else payload
        circuit = cls.model_validate(data)
        circuit.validate_structure()
        return circuit

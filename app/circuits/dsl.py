"""Parser for the Quantum Forge circuit DSL.

Grammar (one statement per line, ``#`` starts a comment)::

    circuit <name>:
    qubits <n>
    bits <n>            # optional, defaults to qubits when measuring
    h q0
    rx(0.5) q1          # or: rx 0.5 q1
    cx q0 q1
    measure all         # or: measure q0 [-> c0]
"""
from __future__ import annotations

import re

from app.circuits.ir import (
    PARAMETRIC_GATES,
    SINGLE_QUBIT_GATES,
    TWO_QUBIT_GATES,
    CircuitIR,
    Instruction,
)


class DSLParseError(ValueError):
    """Raised when DSL source cannot be parsed; carries the line number."""

    def __init__(self, line_no: int, message: str):
        self.line_no = line_no
        super().__init__(f"line {line_no}: {message}")


_QUBIT_RE = re.compile(r"^q(\d+)$")
_CLBIT_RE = re.compile(r"^c(\d+)$")
_PARAM_CALL_RE = re.compile(r"^([a-z]+)\(([^)]*)\)$")


def _parse_qubit(token: str, line_no: int) -> int:
    m = _QUBIT_RE.match(token)
    if not m:
        raise DSLParseError(line_no, f"expected qubit like 'q0', got '{token}'")
    return int(m.group(1))


def _parse_angle(text: str, line_no: int) -> float:
    text = text.strip().replace("pi", "3.141592653589793")
    try:
        # Allow simple expressions like "3.14/2" or "2*1.57".
        if not re.fullmatch(r"[0-9eE+\-.*/() ]+", text):
            raise ValueError
        return float(eval(text, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:
        raise DSLParseError(line_no, f"cannot parse angle '{text}'") from None


def parse_dsl(source: str) -> CircuitIR:
    """Parse DSL source text into a validated :class:`CircuitIR`."""
    name = "circuit"
    num_qubits: int | None = None
    num_clbits = 0
    instructions: list[Instruction] = []
    measured_pairs: list[tuple[int, int]] = []

    for line_no, raw in enumerate(source.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        lowered = line.lower()

        if lowered.startswith("circuit"):
            m = re.match(r"^circuit\s+([\w-]+)\s*:?\s*$", line, re.IGNORECASE)
            if not m:
                raise DSLParseError(line_no, "expected 'circuit <name>:'")
            name = m.group(1)
            continue

        if lowered.startswith("qubits"):
            parts = line.split()
            if len(parts) != 2 or not parts[1].isdigit():
                raise DSLParseError(line_no, "expected 'qubits <n>'")
            num_qubits = int(parts[1])
            continue

        if lowered.startswith("bits"):
            parts = line.split()
            if len(parts) != 2 or not parts[1].isdigit():
                raise DSLParseError(line_no, "expected 'bits <n>'")
            num_clbits = int(parts[1])
            continue

        if num_qubits is None:
            raise DSLParseError(line_no, "declare 'qubits <n>' before instructions")

        tokens = lowered.split()
        head = tokens[0]

        if head == "measure":
            if len(tokens) < 2:
                raise DSLParseError(line_no, "expected 'measure all' or 'measure q<i>'")
            if tokens[1] == "all":
                for q in range(num_qubits):
                    measured_pairs.append((q, q))
                continue
            q = _parse_qubit(tokens[1], line_no)
            if len(tokens) >= 4 and tokens[2] == "->":
                m = _CLBIT_RE.match(tokens[3])
                if not m:
                    raise DSLParseError(line_no, f"expected clbit like 'c0', got '{tokens[3]}'")
                measured_pairs.append((q, int(m.group(1))))
            else:
                measured_pairs.append((q, q))
            continue

        # Gate forms: "rx(0.5) q1" or "rx 0.5 q1" or "h q0" or "cx q0 q1".
        params: list[float] = []
        call = _PARAM_CALL_RE.match(tokens[0])
        if call:
            head = call.group(1)
            params = [_parse_angle(call.group(2), line_no)]
            args = tokens[1:]
        elif head in PARAMETRIC_GATES:
            if len(tokens) < 3:
                raise DSLParseError(line_no, f"'{head}' needs an angle and a qubit")
            params = [_parse_angle(tokens[1], line_no)]
            args = tokens[2:]
        else:
            args = tokens[1:]

        if head in TWO_QUBIT_GATES:
            if len(args) != 2:
                raise DSLParseError(line_no, f"'{head}' needs two qubits")
            qubits = [_parse_qubit(a, line_no) for a in args]
        elif head in SINGLE_QUBIT_GATES:
            if len(args) != 1:
                raise DSLParseError(line_no, f"'{head}' needs one qubit")
            qubits = [_parse_qubit(args[0], line_no)]
        else:
            raise DSLParseError(line_no, f"unknown gate '{head}'")

        instructions.append(Instruction(name=head, qubits=qubits, params=params))

    if num_qubits is None:
        raise DSLParseError(0, "missing 'qubits <n>' declaration")

    if measured_pairs:
        max_clbit = max(c for _, c in measured_pairs)
        num_clbits = max(num_clbits, max_clbit + 1)
        for q, c in measured_pairs:
            instructions.append(Instruction(name="measure", qubits=[q], clbits=[c]))

    circuit = CircuitIR(
        name=name,
        num_qubits=num_qubits,
        num_clbits=num_clbits,
        instructions=instructions,
    )
    circuit.validate_structure()
    return circuit

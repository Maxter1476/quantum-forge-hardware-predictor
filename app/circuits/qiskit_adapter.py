"""Optional Qiskit interoperability.

Qiskit is not a dependency; everything here degrades gracefully when it is
absent. Use :func:`qiskit_available` to check before calling the converters.
"""
from __future__ import annotations

from typing import Any

from app.circuits.ir import CircuitIR, Instruction

try:  # pragma: no cover - depends on environment
    from qiskit import QuantumCircuit  # type: ignore

    _HAS_QISKIT = True
except ImportError:  # pragma: no cover
    QuantumCircuit = None  # type: ignore
    _HAS_QISKIT = False


def qiskit_available() -> bool:
    return _HAS_QISKIT


def to_qiskit(circuit: CircuitIR) -> Any:
    """Convert IR to a qiskit.QuantumCircuit. Requires qiskit installed."""
    if not _HAS_QISKIT:
        raise ImportError("qiskit is not installed; `pip install qiskit` to enable")
    qc = QuantumCircuit(circuit.num_qubits, max(circuit.num_clbits, 1))
    for ins in circuit.instructions:
        if ins.is_measurement:
            qc.measure(ins.qubits[0], ins.clbits[0] if ins.clbits else ins.qubits[0])
        elif ins.name in {"rx", "ry", "rz"}:
            getattr(qc, ins.name)(ins.params[0], ins.qubits[0])
        elif ins.name == "i":
            qc.id(ins.qubits[0])
        elif ins.is_two_qubit:
            getattr(qc, ins.name)(ins.qubits[0], ins.qubits[1])
        else:
            getattr(qc, ins.name)(ins.qubits[0])
    return qc


def from_qiskit(qc: Any, name: str = "imported") -> CircuitIR:
    """Convert a qiskit.QuantumCircuit into IR (supported gates only)."""
    if not _HAS_QISKIT:
        raise ImportError("qiskit is not installed; `pip install qiskit` to enable")
    instructions: list[Instruction] = []
    for item in qc.data:
        op = item.operation
        qubits = [qc.find_bit(q).index for q in item.qubits]
        clbits = [qc.find_bit(c).index for c in item.clbits]
        gate = op.name.lower()
        if gate == "id":
            gate = "i"
        params = [float(p) for p in op.params]
        instructions.append(
            Instruction(name=gate, qubits=qubits, clbits=clbits, params=params)
        )
    circuit = CircuitIR(
        name=name,
        num_qubits=qc.num_qubits,
        num_clbits=qc.num_clbits,
        instructions=instructions,
    )
    circuit.validate_structure()
    return circuit

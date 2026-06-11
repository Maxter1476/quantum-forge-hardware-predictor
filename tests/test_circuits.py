"""Tests for the DSL parser and circuit IR validation."""
import pytest

from app.circuits.dsl import DSLParseError, parse_dsl
from app.circuits.ir import CircuitIR, Instruction


def test_parse_bell(bell_dsl):
    c = parse_dsl(bell_dsl)
    assert c.name == "bell"
    assert c.num_qubits == 2
    assert c.num_clbits == 2
    assert [i.name for i in c.instructions] == ["h", "cx", "measure", "measure"]
    assert c.gate_count == 2
    assert c.two_qubit_gate_count == 1
    assert c.depth == 3  # h, cx, measure layer


def test_parse_parametric_both_syntaxes():
    c = parse_dsl("circuit r:\nqubits 1\nrx(pi/2) q0\nry 0.5 q0\nmeasure all\n")
    rx, ry = c.instructions[0], c.instructions[1]
    assert rx.params[0] == pytest.approx(3.14159265 / 2, rel=1e-6)
    assert ry.params[0] == pytest.approx(0.5)


def test_parse_measure_explicit_clbit():
    c = parse_dsl("circuit m:\nqubits 2\nh q0\nmeasure q0 -> c1\n")
    meas = c.instructions[-1]
    assert meas.qubits == [0] and meas.clbits == [1]
    assert c.num_clbits == 2


def test_parse_errors_carry_line_numbers():
    with pytest.raises(DSLParseError) as exc:
        parse_dsl("circuit bad:\nqubits 2\nfrobnicate q0\n")
    assert "line 3" in str(exc.value)


def test_parse_requires_qubits_declaration():
    with pytest.raises(DSLParseError):
        parse_dsl("circuit bad:\nh q0\n")


def test_ir_validation_rejects_out_of_range_qubit():
    c = CircuitIR(
        num_qubits=2,
        instructions=[Instruction(name="h", qubits=[5])],
    )
    with pytest.raises(ValueError, match="qubit 5"):
        c.validate_structure()


def test_ir_validation_rejects_duplicate_two_qubit_operand():
    c = CircuitIR(
        num_qubits=2,
        instructions=[Instruction(name="cx", qubits=[1, 1])],
    )
    with pytest.raises(ValueError, match="twice"):
        c.validate_structure()


def test_ir_json_roundtrip(bell_dsl):
    c = parse_dsl(bell_dsl)
    again = CircuitIR.from_json(c.to_json())
    assert again == c


def test_cnot_alias_normalized():
    ins = Instruction(name="CNOT", qubits=[0, 1])
    assert ins.name == "cx"


def test_qubit_usage_and_layers(bell_dsl):
    c = parse_dsl(bell_dsl)
    assert c.qubit_usage() == {0: 2, 1: 1}
    layers = c.dependency_layers()
    assert layers[0] == [0]  # h
    assert layers[1] == [1]  # cx depends on h

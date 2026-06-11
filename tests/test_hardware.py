"""Tests for hardware profile loading and the topology estimator."""
import pytest

from app.circuits.dsl import parse_dsl
from app.hardware.presets import all_profiles, get_profile, list_profiles
from app.prediction.transpile import estimate_mapping


def test_all_presets_load_and_are_consistent():
    names = list_profiles()
    assert len(names) == 5
    for profile in all_profiles():
        assert profile.num_qubits >= 2
        assert len(profile.single_qubit_error) == profile.num_qubits
        assert len(profile.readout_error) == profile.num_qubits
        assert len(profile.t1_us) == profile.num_qubits
        for a, b in profile.coupling_map:
            assert 0 <= a < b < profile.num_qubits


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        get_profile("does-not-exist")


def test_profile_edge_helpers():
    p = get_profile("toy-5q")
    assert p.has_edge(0, 1) and p.has_edge(1, 0)
    assert not p.has_edge(0, 4)
    assert p.neighbors(2) == [1, 3]
    assert p.edge_error(1, 2) == pytest.approx(0.012)


def test_bell_maps_natively_on_chain(bell_dsl):
    c = parse_dsl(bell_dsl)
    m = estimate_mapping(c, get_profile("toy-5q"))
    assert m.added_swap_count == 0
    assert m.nonnative_two_qubit_gates == 0
    assert m.hardware_friendly


def test_nonlocal_circuit_requires_swaps():
    # Star interaction graph (q0 talks to everyone): cannot embed in a
    # linear chain where the max degree is 2, so SWAPs are unavoidable.
    src = (
        "circuit star:\nqubits 5\nh q0\ncx q0 q1\ncx q0 q2\ncx q0 q3\ncx q0 q4\n"
        "measure all\n"
    )
    c = parse_dsl(src)
    m = estimate_mapping(c, get_profile("toy-5q"))
    assert m.added_swap_count >= 1
    assert m.added_cnot_count == 3 * m.added_swap_count
    assert m.estimated_depth_after_mapping > c.depth
    assert m.estimated_added_gate_error > 0
    assert m.warnings


def test_circuit_too_large_for_backend():
    src = "circuit big:\nqubits 3\nh q0\nmeasure all\n"
    c = parse_dsl(src)
    with pytest.raises(ValueError, match="only has 2"):
        estimate_mapping(c, get_profile("toy-2q"))

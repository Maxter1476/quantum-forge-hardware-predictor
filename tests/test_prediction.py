"""Tests for the ideal baseline engine and the noise prediction engine."""
import numpy as np
import pytest

from app.circuits.dsl import parse_dsl
from app.hardware.presets import get_profile
from app.prediction.engine import HardwarePredictionEngine
from app.prediction.ideal import IdealEngine
from app.prediction.noise import (
    apply_bit_channel,
    confidence_intervals,
    dist_to_vector,
    mix_with_uniform,
    readout_confusion,
    sample_counts,
)


def test_ideal_bell_distribution(bell_dsl):
    c = parse_dsl(bell_dsl)
    probs = IdealEngine().measured_probabilities(c)
    assert set(probs) == {"00", "11"}
    assert probs["00"] == pytest.approx(0.5)
    assert probs["11"] == pytest.approx(0.5)


def test_ideal_x_gate_flips_bit():
    c = parse_dsl("circuit flip:\nqubits 2\nx q1\nmeasure all\n")
    probs = IdealEngine().measured_probabilities(c)
    assert probs == {"01": pytest.approx(1.0)}


def test_ideal_ghz():
    c = parse_dsl(
        "circuit ghz:\nqubits 3\nh q0\ncx q0 q1\ncx q1 q2\nmeasure all\n"
    )
    probs = IdealEngine().measured_probabilities(c)
    assert set(probs) == {"000", "111"}


def test_ideal_engine_qubit_limit():
    c = parse_dsl("circuit big:\nqubits 13\nh q0\nmeasure all\n")
    with pytest.raises(ValueError, match="supports up to"):
        IdealEngine(max_qubits=12).statevector(c)


def test_readout_transformation_flips_known_state():
    # Deterministic |0>: confusion matrix moves exactly e mass to '1'.
    vec = dist_to_vector({"0": 1.0}, 1)
    out = apply_bit_channel(vec, 1, 0, readout_confusion(0.1, 0.2))
    assert out[0] == pytest.approx(0.9)
    assert out[1] == pytest.approx(0.1)


def test_mix_with_uniform_preserves_normalization():
    vec = dist_to_vector({"00": 0.7, "11": 0.3}, 2)
    out = mix_with_uniform(vec, 0.2)
    assert out.sum() == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.05)  # pure uniform share


def test_gate_error_accumulation_lowers_reliability(bell_dsl):
    c = parse_dsl(bell_dsl)
    engine = HardwarePredictionEngine()
    clean = engine.predict(c, get_profile("pristine-8q"), shots=1024, seed=1)
    noisy = engine.predict(c, get_profile("noisy-edu-4q"), shots=1024, seed=1)
    assert noisy.reliability_score < clean.reliability_score
    assert noisy.error_contributions["gate_error"] > clean.error_contributions["gate_error"]


def test_predicted_probabilities_normalized_and_leak_to_other_bitstrings(bell_dsl):
    c = parse_dsl(bell_dsl)
    r = HardwarePredictionEngine().predict(c, get_profile("toy-5q"), shots=4096, seed=3)
    total = sum(r.predicted_probabilities.values())
    assert total == pytest.approx(1.0, abs=1e-9)
    # Noise must populate the ideally-forbidden outcomes.
    assert r.predicted_probabilities.get("01", 0) > 0
    assert r.predicted_probabilities.get("10", 0) > 0
    # But the ideal peaks should still dominate.
    assert r.predicted_probabilities["00"] > 0.35
    assert r.predicted_probabilities["11"] > 0.35


def test_confidence_intervals_bound_predictions():
    cis = confidence_intervals({"00": 0.5, "11": 0.5}, shots=1000)
    lo, hi = cis["00"]
    assert 0 <= lo < 0.5 < hi <= 1.0
    wide = confidence_intervals({"00": 0.5}, shots=1000, widen_factor=2.0)
    assert wide["00"][1] - wide["00"][0] > hi - lo


def test_sampled_counts_sum_to_shots(bell_dsl):
    counts = sample_counts({"00": 0.5, "11": 0.5}, shots=777, seed=0)
    assert sum(counts.values()) == 777


def test_prediction_is_seed_reproducible(bell_dsl):
    c = parse_dsl(bell_dsl)
    p = get_profile("toy-2q")
    engine = HardwarePredictionEngine()
    a = engine.predict(c, p, shots=512, seed=42)
    b = engine.predict(c, p, shots=512, seed=42)
    assert a.predicted_counts == b.predicted_counts
    # Drift depends on wall-clock calibration age, so allow float-level slack.
    for k, v in a.predicted_probabilities.items():
        assert b.predicted_probabilities[k] == pytest.approx(v, abs=1e-6)


def test_decoherence_biases_toward_ground_state():
    # A long idle-heavy circuit holding |1> should relax toward |0>.
    src = "circuit hold:\nqubits 1\nx q0\n" + "i q0\n" * 30 + "measure all\n"
    c = parse_dsl(src)
    r = HardwarePredictionEngine().predict(c, get_profile("noisy-edu-4q"), shots=1024, seed=5)
    assert r.predicted_probabilities.get("0", 0) > 0.01
    assert r.error_contributions["decoherence"] > 0


def test_error_contribution_keys_complete(bell_dsl):
    c = parse_dsl(bell_dsl)
    r = HardwarePredictionEngine().predict(c, get_profile("toy-5q"), shots=256, seed=0)
    assert set(r.error_contributions) == {
        "gate_error",
        "topology",
        "crosstalk",
        "decoherence",
        "leakage",
        "readout",
        "drift",
    }
    assert r.dominant_error_sources

"""Tests for synthetic dataset generation and the correction model."""
import numpy as np
import pytest

from app.hardware.presets import get_profile
from app.learning.correction import CorrectionModel, _apply_weight, optimal_weight
from app.learning.features import FEATURE_NAMES, extract_features
from app.learning.synthetic import generate_dataset, perturb_profile, random_circuit
from app.prediction.transpile import estimate_mapping
import random


def test_random_circuit_is_valid():
    rng = random.Random(0)
    c = random_circuit(rng, 3, 8)
    c.validate_structure()
    assert c.num_qubits == 3
    assert c.measured_qubits == [0, 1, 2]


def test_perturb_profile_scales_errors():
    base = get_profile("toy-5q")
    hidden, scale = perturb_profile(base, random.Random(1))
    assert scale != 1.0
    assert hidden.single_qubit_error[0] == pytest.approx(
        min(0.4, base.single_qubit_error[0] * scale)
    )
    assert hidden.t1_us[0] == pytest.approx(base.t1_us[0] / scale)


def test_generate_dataset_shapes():
    examples = generate_dataset(get_profile("toy-5q"), num_examples=6, shots=128, seed=2)
    assert len(examples) == 6
    for ex in examples:
        assert sum(ex.true_counts.values()) == 128
        assert ex.hidden_scale > 0


def test_feature_vector_matches_names():
    profile = get_profile("toy-5q")
    c = random_circuit(random.Random(3), 3, 6)
    mapping = estimate_mapping(c, profile)
    feats = extract_features(c, profile, mapping)
    assert feats.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(feats).all()


def test_optimal_weight_recovers_known_mixing():
    pred = np.array([0.7, 0.1, 0.1, 0.1])
    truth = _apply_weight(pred, 0.3)
    w = optimal_weight(pred, truth)
    assert w == pytest.approx(0.3, abs=0.02)


def test_correction_model_trains_and_roundtrips(tmp_path):
    profile = get_profile("noisy-edu-4q")
    examples = generate_dataset(profile, num_examples=10, shots=256, seed=4)
    model = CorrectionModel("ridge")
    metrics = model.train(examples)
    assert metrics.num_examples == 10
    assert metrics.mean_tvd_after <= metrics.mean_tvd_before + 1e-9

    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = CorrectionModel.load(path)
    assert loaded.trained
    assert loaded.model_type == "ridge"

    c = random_circuit(random.Random(5), 3, 5)
    mapping = estimate_mapping(c, profile)
    w = loaded.predict_weight(c, profile, mapping)
    assert -0.4 <= w <= 0.6


def test_untrained_model_refuses_to_predict():
    model = CorrectionModel("ridge")
    with pytest.raises(RuntimeError):
        model.predict_weight(None, None, None)

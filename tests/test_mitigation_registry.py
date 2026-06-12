"""Tests for readout mitigation, the profile registry, and new endpoints."""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.circuits.dsl import parse_dsl
from app.hardware import registry
from app.hardware.presets import get_profile
from app.prediction.engine import HardwarePredictionEngine
from app.prediction.mitigation import mitigate_readout
from app.utils import service


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _isolated_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "PROFILES_DIR", tmp_path / "profiles")


def _custom_profile_data(name="lab-3q"):
    return {
        "backend_name": name,
        "num_qubits": 3,
        "coupling_map": [[0, 1], [1, 2]],
        "single_qubit_error": {0: 0.001, 1: 0.002, 2: 0.001},
        "two_qubit_error": {"0-1": 0.01, "1-2": 0.02},
        "readout_error": {0: 0.02, 1: 0.03, 2: 0.02},
        "t1_us": {0: 100, 1: 90, 2: 110},
        "t2_us": {0: 80, 1: 70, 2: 85},
        "description": "transcribed lab calibration",
    }


# -- readout mitigation -------------------------------------------------------


def test_mitigation_moves_prediction_toward_ideal(bell_dsl):
    c = parse_dsl(bell_dsl)
    p = get_profile("noisy-edu-4q")  # 6% readout error: mitigation matters
    result = HardwarePredictionEngine().predict(c, p, shots=2048, seed=9)
    mit = mitigate_readout(result, p)
    assert mit["tvd_to_ideal_after"] < mit["tvd_to_ideal_before"]
    assert mit["improvement"] > 0
    probs = mit["mitigated_probabilities"]
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)
    # The ideally-forbidden outcomes should shrink after mitigation.
    assert probs.get("01", 0) < result.predicted_probabilities.get("01", 1)


def test_mitigation_via_service_payload(bell_dsl):
    c = parse_dsl(bell_dsl)
    p = get_profile("toy-5q")
    payload = service.run_prediction(
        c, p, shots=512, seed=1, persist=False, apply_readout_mitigation=True
    )
    assert "readout_mitigation" in payload
    assert payload["readout_mitigation"]["improvement"] > 0


# -- profile registry ---------------------------------------------------------


def test_register_and_get_custom_profile():
    registry.register_profile(_custom_profile_data())
    assert "lab-3q" in registry.list_profiles()
    loaded = registry.get_profile("lab-3q")
    assert loaded.num_qubits == 3
    assert loaded.has_edge(0, 1)


def test_register_rejects_preset_name_and_duplicates():
    with pytest.raises(ValueError, match="built-in preset"):
        registry.register_profile(_custom_profile_data(name="toy-5q"))
    registry.register_profile(_custom_profile_data())
    with pytest.raises(ValueError, match="already exists"):
        registry.register_profile(_custom_profile_data())
    registry.register_profile(_custom_profile_data(), overwrite=True)


def test_register_rejects_incomplete_maps():
    data = _custom_profile_data()
    data["readout_error"].pop(2)
    with pytest.raises(ValueError, match="readout_error missing"):
        registry.register_profile(data)


def test_delete_custom_profile():
    registry.register_profile(_custom_profile_data())
    assert registry.delete_custom_profile("lab-3q")
    assert not registry.delete_custom_profile("lab-3q")
    with pytest.raises(KeyError):
        registry.get_profile("lab-3q")


# -- API ----------------------------------------------------------------------


def test_api_register_profile_and_predict_on_it(client, bell_dsl):
    resp = client.post(
        "/hardware-profiles", json={"profile": _custom_profile_data()}
    )
    assert resp.status_code == 201
    assert resp.json()["registered"] == "lab-3q"

    resp = client.get("/hardware-profiles/lab-3q")
    assert resp.status_code == 200

    resp = client.post(
        "/predict", json={"dsl": bell_dsl, "backend": "lab-3q", "shots": 128}
    )
    assert resp.status_code == 200


def test_api_register_invalid_profile(client):
    data = _custom_profile_data()
    data["backend_name"] = "!!"
    resp = client.post("/hardware-profiles", json={"profile": data})
    assert resp.status_code == 422


def test_api_predict_with_mitigation(client, bell_dsl):
    resp = client.post(
        "/predict",
        json={
            "dsl": bell_dsl,
            "backend": "toy-5q",
            "shots": 256,
            "seed": 4,
            "apply_readout_mitigation": True,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["readout_mitigation"]["improvement"] > 0


def test_api_compare_backends(client, bell_dsl):
    resp = client.post(
        "/predict/compare",
        json={
            "dsl": bell_dsl,
            "backends": ["toy-2q", "noisy-edu-4q", "pristine-8q"],
            "shots": 256,
            "seed": 2,
        },
    )
    assert resp.status_code == 200
    rows = resp.json()["comparison"]
    assert [r["backend"] for r in rows][0] == "pristine-8q"  # sorted by reliability
    scores = [r["reliability_score"] for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_api_compare_handles_too_small_backend(client):
    big = "circuit big:\nqubits 4\nh q0\ncx q0 q1\ncx q1 q2\ncx q2 q3\nmeasure all\n"
    resp = client.post(
        "/predict/compare",
        json={"dsl": big, "backends": ["toy-2q", "toy-5q"], "shots": 64},
    )
    assert resp.status_code == 200
    rows = {r["backend"]: r for r in resp.json()["comparison"]}
    assert "error" in rows["toy-2q"]
    assert "reliability_score" in rows["toy-5q"]

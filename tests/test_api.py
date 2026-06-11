"""API endpoint tests using FastAPI's TestClient."""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_hardware_profiles(client):
    resp = client.get("/hardware-profiles")
    assert resp.status_code == 200
    names = [p["backend_name"] for p in resp.json()["profiles"]]
    assert "toy-2q" in names and "heavyhex-12q-mock" in names


def test_get_profile_and_404(client):
    assert client.get("/hardware-profiles/toy-5q").status_code == 200
    assert client.get("/hardware-profiles/nope").status_code == 404


def test_parse_endpoint(client, bell_dsl):
    resp = client.post("/circuits/parse", json={"dsl": bell_dsl})
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["num_qubits"] == 2
    assert body["ir"]["instructions"][0]["name"] == "h"


def test_parse_endpoint_rejects_bad_dsl(client):
    resp = client.post("/circuits/parse", json={"dsl": "circuit x:\nqubits 1\nzap q0\n"})
    assert resp.status_code == 422


def test_predict_endpoint(client, bell_dsl):
    resp = client.post(
        "/predict",
        json={"dsl": bell_dsl, "backend": "toy-5q", "shots": 256, "seed": 1},
    )
    assert resp.status_code == 200
    body = resp.json()
    result = body["result"]
    assert sum(result["predicted_counts"].values()) == 256
    assert 0 <= result["reliability_score"] <= 100
    assert body["council"]["reliability_classification"] in {"high", "moderate", "low"}
    assert body["run_id"]


def test_predict_report_and_fetch(client, bell_dsl):
    resp = client.post(
        "/predict/report",
        json={"dsl": bell_dsl, "backend": "toy-2q", "shots": 128, "seed": 2},
    )
    assert resp.status_code == 200
    report_id = resp.json()["report_id"]
    assert report_id

    listing = client.get("/reports").json()["reports"]
    assert any(r["report_id"] == report_id for r in listing)

    fetched = client.get(f"/reports/{report_id}")
    assert fetched.status_code == 200
    assert "# Hardware Prediction Report" in fetched.json()["markdown"]

    assert client.get("/reports/missing").status_code == 404


def test_examples_endpoint(client):
    resp = client.get("/examples")
    assert resp.status_code == 200
    assert "bell" in resp.json()["examples"]


def test_train_synthetic_endpoint(client):
    resp = client.post(
        "/train/synthetic",
        json={
            "backend": "toy-2q",
            "num_examples": 6,
            "shots": 64,
            "model_type": "ridge",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["metrics"]["num_examples"] == 6
    assert body["model_path"].endswith(".pkl")

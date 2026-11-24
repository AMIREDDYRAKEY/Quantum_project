"""Integration tests for Flask REST API Endpoints (Phase 11)"""
import pytest
from backend.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "PennyLane" in data["quantum_framework"]


def test_list_datasets(client):
    res = client.get("/api/datasets/list")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert len(data["datasets"]) == 4


def test_dataset_summary(client):
    res = client.get("/api/datasets/iris/summary")
    assert res.status_code == 200
    data = res.get_json()
    assert data["summary"]["id"] == "iris"
    assert "class_distribution" in data["summary"]


def test_classical_knn_endpoint(client):
    res = client.post("/api/classical/train", json={"dataset": "iris", "k": 3})
    assert res.status_code == 200
    data = res.get_json()
    assert "results" in data
    assert data["results"]["accuracy"] > 0.8


def test_quantum_circuit_endpoint(client):
    res = client.get("/api/quantum/circuit?feature_map=hybrid&n_qubits=4&depth=1")
    assert res.status_code == 200
    data = res.get_json()
    assert "ascii_diagram" in data
    assert "metadata" in data


def test_quantum_train_endpoint(client):
    res = client.post(
        "/api/quantum/train",
        json={"dataset": "iris", "feature_map": "z", "k": 3, "subset_size": 25},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert "results" in data
    assert "accuracy" in data["results"]


def test_benchmarks_paper_results(client):
    res = client.get("/api/benchmarks/paper-results")
    assert res.status_code == 200
    data = res.get_json()
    assert "table_iii_results" in data["data"]


def test_benchmarks_live_results(client):
    res = client.get("/api/benchmarks/live-results")
    assert res.status_code == 200
    data = res.get_json()
    assert "iris" in data["data"]


def test_predict_single_endpoint(client):
    res = client.post(
        "/api/predict/single",
        json={
            "dataset": "iris",
            "model_type": "classical_knn",
            "features": [5.1, 3.5, 1.4, 0.2],
        },
    )
    assert res.status_code == 200
    data = res.get_json()
    assert "predicted_class" in data
    assert "predicted_label" in data

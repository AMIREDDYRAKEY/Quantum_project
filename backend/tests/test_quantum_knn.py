"""Unit tests for Quantum KNN Classifier (Phase 6)"""
import pytest
import numpy as np
from backend.core.quantum_knn import QuantumKNNClassifier
from backend.core.data_manager import DataManager


@pytest.fixture
def sample_data():
    dm = DataManager(random_state=42)
    data = dm.split_and_preprocess("iris")
    # Take a small fast slice for unit testing
    return {
        "X_train": data["X_train_angles"][:20],
        "y_train": data["y_train"][:20],
        "X_test": data["X_test_angles"][:5],
        "y_test": data["y_test"][:5],
        "class_labels": data["class_labels"],
    }


@pytest.mark.parametrize("fmap", ["z", "zz", "pauli", "hybrid"])
def test_qknn_all_feature_maps(sample_data, fmap):
    qknn = QuantumKNNClassifier(feature_map=fmap, n_neighbors=3, n_wires=4)
    qknn.fit(sample_data["X_train"], sample_data["y_train"])
    preds = qknn.predict(sample_data["X_test"])
    assert len(preds) == len(sample_data["y_test"])


def test_qknn_neighbor_details(sample_data):
    qknn = QuantumKNNClassifier(feature_map="hybrid", n_neighbors=3, n_wires=4)
    qknn.fit(sample_data["X_train"], sample_data["y_train"])
    details = qknn.get_neighbor_details(sample_data["X_test"][0])
    assert details["k"] == 3
    assert len(details["neighbors"]) == 3
    for n in details["neighbors"]:
        assert 0.0 <= n["fidelity"] <= 1.0
        assert 0.0 <= n["quantum_distance"] <= 1.0


def test_qknn_evaluate(sample_data):
    qknn = QuantumKNNClassifier(feature_map="hybrid", n_neighbors=3, n_wires=4)
    results = qknn.evaluate(
        sample_data["X_train"],
        sample_data["y_train"],
        sample_data["X_test"],
        sample_data["y_test"],
    )
    assert 0.0 <= results["accuracy"] <= 1.0
    assert "confusion_matrix" in results

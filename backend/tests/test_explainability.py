"""Unit tests for Explainable AI (XAI) Suite (Phase 8)"""
import pytest
import numpy as np
from backend.core.explainability import ExplainabilityEngine
from backend.core.classical_knn import ClassicalKNNBaseline
from backend.core.quantum_knn import QuantumKNNClassifier
from backend.core.data_manager import DataManager


@pytest.fixture
def dataset_slice():
    dm = DataManager(random_state=42)
    data = dm.split_and_preprocess("iris")
    return {
        "X_train": data["X_train"][:30],
        "X_test": data["X_test"][:10],
        "X_train_angles": data["X_train_angles"][:30],
        "X_test_angles": data["X_test_angles"][:10],
        "y_train": data["y_train"][:30],
        "y_test": data["y_test"][:10],
        "feature_names": data["selected_features"],
        "class_labels": data["class_labels"],
    }


def test_classical_shap_explanations(dataset_slice):
    model = ClassicalKNNBaseline(n_neighbors=3)
    model.fit(dataset_slice["X_train"], dataset_slice["y_train"])

    sample = dataset_slice["X_test"][0]
    explanation = ExplainabilityEngine.explain_classical_predictions(
        model,
        dataset_slice["X_train"],
        sample,
        dataset_slice["feature_names"],
        max_background=15,
    )
    assert "feature_importance" in explanation
    assert len(explanation["feature_importance"]) == len(dataset_slice["feature_names"])
    for item in explanation["feature_importance"]:
        assert "feature" in item
        assert "importance" in item


def test_quantum_sensitivity_gradients(dataset_slice):
    qknn = QuantumKNNClassifier(feature_map="hybrid", n_neighbors=3, n_wires=4)
    qknn.fit(dataset_slice["X_train_angles"], dataset_slice["y_train"])

    sample = dataset_slice["X_test_angles"][0]
    sensitivities = ExplainabilityEngine.compute_quantum_sensitivity(
        qknn.engine, sample, dataset_slice["X_train_angles"], dataset_slice["feature_names"]
    )
    assert len(sensitivities) == len(dataset_slice["feature_names"])
    for s in sensitivities:
        assert s["sensitivity"] >= 0.0


def test_quantum_prediction_decomposition(dataset_slice):
    qknn = QuantumKNNClassifier(feature_map="hybrid", n_neighbors=3, n_wires=4)
    qknn.fit(dataset_slice["X_train_angles"], dataset_slice["y_train"])

    sample = dataset_slice["X_test_angles"][0]
    explanation = ExplainabilityEngine.explain_quantum_prediction(
        qknn, sample, dataset_slice["feature_names"], dataset_slice["class_labels"]
    )
    assert "neighbor_contributions" in explanation
    assert len(explanation["neighbor_contributions"]) == 3
    assert "class_votes" in explanation
    assert "quantum_feature_sensitivities" in explanation


def test_feature_maps_comparison_telemetry():
    telemetry = ExplainabilityEngine.get_feature_maps_comparison_telemetry(n_qubits=4, depth=1)
    assert len(telemetry) == 4
    map_names = [t["feature_map"] for t in telemetry]
    assert set(map_names) == {"z", "zz", "pauli", "hybrid"}

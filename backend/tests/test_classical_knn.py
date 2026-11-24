"""Unit tests for Classical KNN Baseline Classifier (Phase 4)"""
import pytest
import numpy as np
from backend.core.classical_knn import ClassicalKNNBaseline
from backend.core.data_manager import DataManager


@pytest.fixture
def iris_data():
    dm = DataManager(random_state=42)
    return dm.split_and_preprocess("iris")


def test_classical_knn_fit_predict(iris_data):
    model = ClassicalKNNBaseline(n_neighbors=3, metric="euclidean")
    model.fit(iris_data["X_train"], iris_data["y_train"])
    preds = model.predict(iris_data["X_test"])
    assert len(preds) == len(iris_data["y_test"])
    assert set(preds).issubset(set(iris_data["classes"]))


def test_classical_knn_evaluate_metrics(iris_data):
    model = ClassicalKNNBaseline(n_neighbors=3, metric="euclidean")
    results = model.evaluate(
        iris_data["X_train"],
        iris_data["y_train"],
        iris_data["X_test"],
        iris_data["y_test"],
        class_labels=iris_data["class_labels"],
    )
    assert 0.0 <= results["accuracy"] <= 1.0
    assert 0.0 <= results["precision"] <= 1.0
    assert 0.0 <= results["recall"] <= 1.0
    assert 0.0 <= results["f1_score"] <= 1.0
    assert len(results["confusion_matrix"]) == 3
    assert results["training_time_ms"] >= 0.0


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
def test_classical_knn_distance_metrics(iris_data, metric):
    model = ClassicalKNNBaseline(n_neighbors=3, metric=metric)
    results = model.evaluate(
        iris_data["X_train"],
        iris_data["y_train"],
        iris_data["X_test"],
        iris_data["y_test"],
    )
    assert results["accuracy"] > 0.8

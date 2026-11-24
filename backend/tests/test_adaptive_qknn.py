"""Unit tests for Adaptive Hybrid Quantum KNN (Phase 7 - Proposed Model)"""
import pytest
import numpy as np
from backend.core.adaptive_hybrid_qknn import (
    AdaptiveHybridQKNN,
    run_full_benchmark_comparison,
)
from backend.core.data_manager import DataManager


@pytest.fixture
def fast_dataset():
    dm = DataManager(random_state=42)
    # Use Iris with 30 train samples and 10 test samples for fast unit tests
    data = dm.split_and_preprocess("iris")
    return {
        "X_train": data["X_train"][:30],
        "X_test": data["X_test"][:10],
        "X_train_angles": data["X_train_angles"][:30],
        "X_test_angles": data["X_test_angles"][:10],
        "y_train": data["y_train"][:30],
        "y_test": data["y_test"][:10],
        "class_labels": data["class_labels"],
    }


def test_adaptive_hybrid_fit_and_selection(fast_dataset):
    model = AdaptiveHybridQKNN(
        candidate_feature_maps=["z", "hybrid"],
        candidate_k=[1, 3],
        weighted=True,
        n_wires=4,
        cv_folds=2,
        random_state=42,
    )
    model.fit(fast_dataset["X_train_angles"], fast_dataset["y_train"])
    assert model.is_fitted is True
    assert model.best_feature_map in ["z", "hybrid"]
    assert model.best_k in [1, 3]
    assert len(model.tuning_log_) > 0

    preds = model.predict(fast_dataset["X_test_angles"])
    assert len(preds) == len(fast_dataset["y_test"])


def test_full_benchmark_comparison(fast_dataset):
    comparison = run_full_benchmark_comparison(
        fast_dataset, k=3, candidate_k=[1, 3], n_wires=4
    )
    # Check that all 6 models are present
    expected_models = [
        "classical_knn",
        "z_qknn",
        "zz_qknn",
        "pauli_qknn",
        "hybrid_qknn",
        "adaptive_hybrid_qknn",
        "summary_analysis",
    ]
    for m in expected_models:
        assert m in comparison, f"Model {m} missing from comparison"

    assert "accuracy" in comparison["adaptive_hybrid_qknn"]
    assert "f1_score" in comparison["adaptive_hybrid_qknn"]
    assert "improvement_detected" in comparison["summary_analysis"]

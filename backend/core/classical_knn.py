"""
Classical K-Nearest Neighbour Baseline Classifier (Phase 4)
Uses Scikit-learn KNeighborsClassifier with full metrics, confusion matrix,
and configurable distance metrics (Euclidean, Manhattan, Chebyshev).
"""
import time
from typing import Dict, Any, Optional
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class ClassicalKNNBaseline:
    """
    Configurable Classical KNN Baseline model for comparative benchmarking.
    """

    def __init__(
        self,
        n_neighbors: int = 3,
        metric: str = "euclidean",
        weights: str = "uniform",
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            weights=weights,
        )
        self.is_fitted = False
        self.training_time_ms = 0.0
        self.inference_time_ms = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit classical KNN model."""
        start_time = time.perf_counter()
        self.model.fit(X_train, y_train)
        self.training_time_ms = (time.perf_counter() - start_time) * 1000.0
        self.is_fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels for query instances."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        start_time = time.perf_counter()
        preds = self.model.predict(X_test)
        self.inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        return preds

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict_proba(X_test)

    def kneighbors(self, X_test: np.ndarray, n_neighbors: Optional[int] = None):
        """Find the K-neighbors of a point."""
        k = n_neighbors or self.n_neighbors
        return self.model.kneighbors(X_test, n_neighbors=k)

    def evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_labels: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Train and evaluate the baseline model with full research metrics."""
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        labels = class_labels or [str(c) for c in sorted(np.unique(y_test))]

        return {
            "model_type": "Classical_KNN",
            "k": self.n_neighbors,
            "metric": self.metric,
            "weights": self.weights,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "class_labels": labels,
            "classification_report": report,
            "training_time_ms": round(self.training_time_ms, 2),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        }

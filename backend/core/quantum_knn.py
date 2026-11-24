"""
Quantum K-Nearest Neighbour Classifier (Phase 6)
Implements QKNN utilizing PennyLane quantum circuits and state-inversion fidelity.
Supports independent feature maps: Z, ZZ, Pauli, and Hybrid.
Implements standard unweighted majority voting as baseline.
"""
import time
from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from backend.core.quantum_similarity import QuantumSimilarityEngine


class QuantumKNNClassifier:
    """
    Quantum K-Nearest Neighbour Classifier:
    1. Encodes classical samples into quantum states using selected feature map.
    2. Computes quantum state fidelities using inversion overlap circuits.
    3. Derives quantum distance D_Q = 1 - Fidelity.
    4. Selects K nearest quantum neighbors and performs majority voting.
    """

    def __init__(
        self,
        feature_map: str = "hybrid",
        n_neighbors: int = 3,
        n_wires: int = 4,
        depth: int = 1,
    ):
        self.feature_map = feature_map.lower().strip()
        self.n_neighbors = n_neighbors
        self.n_wires = n_wires
        self.depth = depth

        self.engine = QuantumSimilarityEngine(
            feature_map_name=self.feature_map,
            n_wires=self.n_wires,
            depth=self.depth,
        )

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.states_train: Optional[np.ndarray] = None
        self.classes: List[int] = []
        self.is_fitted = False
        self.training_time_ms = 0.0
        self.inference_time_ms = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit QKNN by encoding training samples into quantum state vectors."""
        start_time = time.perf_counter()
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=int)
        self.classes = sorted(np.unique(self.y_train).tolist())

        # Precompute quantum states for all training instances
        self.states_train = self.engine.compute_statevectors(self.X_train)
        self.training_time_ms = (time.perf_counter() - start_time) * 1000.0
        self.is_fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels for test instances using quantum majority voting."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        start_time = time.perf_counter()
        X_test = np.asarray(X_test, dtype=np.float64)

        # Pairwise fidelities and quantum distances
        _, distances = self.engine.compute_fidelity_matrix(
            X_test, self.X_train, states_train=self.states_train
        )

        n_queries = len(X_test)
        predictions = np.zeros(n_queries, dtype=int)

        for i in range(n_queries):
            dists = distances[i]
            # Indices of k smallest distances
            k_indices = np.argsort(dists)[: self.n_neighbors]
            k_labels = self.y_train[k_indices]

            # Majority voting
            unique_labels, counts = np.unique(k_labels, return_counts=True)
            winner = unique_labels[np.argmax(counts)]
            predictions[i] = winner

        self.inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        return predictions

    def get_neighbor_details(self, x_query: np.ndarray) -> Dict[str, Any]:
        """
        Extract detailed neighborhood info for a single query:
        neighbor indices, labels, quantum fidelities, and quantum distances.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        x_query = np.asarray(x_query, dtype=np.float64).reshape(1, -1)
        fidelities, distances = self.engine.compute_fidelity_matrix(
            x_query, self.X_train, states_train=self.states_train
        )

        fids = fidelities[0]
        dists = distances[0]
        k_indices = np.argsort(dists)[: self.n_neighbors]

        neighbors = []
        for rank, idx in enumerate(k_indices):
            neighbors.append({
                "rank": rank + 1,
                "train_index": int(idx),
                "label": int(self.y_train[idx]),
                "fidelity": float(fids[idx]),
                "quantum_distance": float(dists[idx]),
            })

        return {
            "query_features": x_query[0].tolist(),
            "k": self.n_neighbors,
            "feature_map": self.feature_map,
            "neighbors": neighbors,
        }

    def evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_labels: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Train and evaluate the Quantum KNN model with full research metrics."""
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
            "model_type": f"Quantum_KNN_{self.feature_map.upper()}",
            "feature_map": self.feature_map,
            "k": self.n_neighbors,
            "n_wires": self.n_wires,
            "depth": self.depth,
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

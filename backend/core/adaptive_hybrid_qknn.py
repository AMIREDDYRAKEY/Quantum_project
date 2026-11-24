"""
Adaptive Hybrid Quantum K-Nearest Neighbour Classifier (Phase 7 - Proposed Model)
Implements:
1. Validation-guided Adaptive Feature Map Selection (Z vs ZZ vs Pauli vs Hybrid)
2. Validation-based Optimal K Selection (k in candidate set, e.g. 1, 3, 5, 7, 9)
3. Fidelity-weighted and Inverse-Distance weighted Quantum Nearest-Neighbor voting
4. Strictly held-out test evaluation (Zero leakage of test data during tuning)
5. Comprehensive benchmarking harness comparing Classical KNN, Z-QKNN, ZZ-QKNN,
   Pauli-QKNN, Hybrid-QKNN, and AH-QKNN.
"""
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from backend.core.quantum_knn import QuantumKNNClassifier
from backend.core.quantum_similarity import QuantumSimilarityEngine
from backend.core.classical_knn import ClassicalKNNBaseline


class AdaptiveHybridQKNN:
    """
    Adaptive Hybrid Quantum KNN (AH-QKNN):
    - Automatically discovers the optimal quantum feature map and optimal K
      using internal cross-validation strictly on the training partition.
    - Implements fidelity-weighted quantum voting to emphasize high-confidence neighbors.
    """

    def __init__(
        self,
        candidate_feature_maps: Optional[List[str]] = None,
        candidate_k: Optional[List[int]] = None,
        weighted: bool = True,
        weighting_scheme: str = "fidelity",  # 'fidelity' or 'inverse_distance'
        n_wires: int = 4,
        depth: int = 1,
        cv_folds: int = 3,
        random_state: int = 42,
    ):
        self.candidate_feature_maps = candidate_feature_maps or ["z", "zz", "pauli", "hybrid"]
        self.candidate_k = candidate_k or [1, 3, 5, 7]
        self.weighted = weighted
        self.weighting_scheme = weighting_scheme
        self.n_wires = n_wires
        self.depth = depth
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Optimal parameters determined during fit
        self.best_feature_map: Optional[str] = None
        self.best_k: int = 3
        self.cv_results_: Dict[str, Any] = {}
        self.tuning_log_: List[Dict[str, Any]] = []

        self.engine_: Optional[QuantumSimilarityEngine] = None
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.states_train_: Optional[np.ndarray] = None
        self.classes_: List[int] = []
        self.is_fitted = False
        self.training_time_ms = 0.0
        self.inference_time_ms = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fits the AH-QKNN model:
        1. Performs Stratified K-Fold CV strictly on X_train to evaluate all (feature_map, k) combinations.
        2. Selects the combination with the highest mean validation F1-score.
        3. Precomputes quantum state vectors for the selected optimal feature map.
        """
        start_time = time.perf_counter()
        self.X_train_ = np.asarray(X_train, dtype=np.float64)
        self.y_train_ = np.asarray(y_train, dtype=int)
        self.classes_ = sorted(np.unique(self.y_train_).tolist())

        skf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        tuning_records = []
        best_score = -1.0
        best_map = self.candidate_feature_maps[0]
        best_k_val = self.candidate_k[0]

        # Evaluate candidate feature maps
        for fmap in self.candidate_feature_maps:
            engine = QuantumSimilarityEngine(
                feature_map_name=fmap,
                n_wires=self.n_wires,
                depth=self.depth,
            )

            # Precompute states for entire training set to speed up fold evaluations
            all_states = engine.compute_statevectors(self.X_train_)

            for k in self.candidate_k:
                fold_scores = []
                for train_idx, val_idx in skf.split(self.X_train_, self.y_train_):
                    # In-fold train & validation states
                    val_states = all_states[val_idx]
                    fold_train_states = all_states[train_idx]
                    y_val = self.y_train_[val_idx]
                    fold_y_train = self.y_train_[train_idx]

                    # Inner products
                    inner = np.matmul(val_states, fold_train_states.conj().T)
                    fids = np.clip(np.abs(inner) ** 2, 0.0, 1.0)
                    dists = np.clip(1.0 - fids, 0.0, 1.0)

                    # Predict validation fold
                    val_preds = self._predict_from_distances(
                        dists, fids, fold_y_train, k, self.weighted, self.weighting_scheme
                    )
                    fold_f1 = float(f1_score(y_val, val_preds, average="weighted", zero_division=0))
                    fold_scores.append(fold_f1)

                mean_f1 = float(np.mean(fold_scores))
                record = {
                    "feature_map": fmap,
                    "k": k,
                    "weighted": self.weighted,
                    "mean_cv_f1": round(mean_f1, 4),
                    "std_cv_f1": round(float(np.std(fold_scores)), 4),
                }
                tuning_records.append(record)

                if mean_f1 > best_score:
                    best_score = mean_f1
                    best_map = fmap
                    best_k_val = k

        self.best_feature_map = best_map
        self.best_k = best_k_val
        self.tuning_log_ = tuning_records
        self.cv_results_ = {
            "best_feature_map": best_map,
            "best_k": best_k_val,
            "best_cv_score": round(best_score, 4),
            "total_configurations_tested": len(tuning_records),
        }

        # Initialize the optimal engine and compute full train statevectors
        self.engine_ = QuantumSimilarityEngine(
            feature_map_name=self.best_feature_map,
            n_wires=self.n_wires,
            depth=self.depth,
        )
        self.states_train_ = self.engine_.compute_statevectors(self.X_train_)

        self.training_time_ms = (time.perf_counter() - start_time) * 1000.0
        self.is_fitted = True
        return self

    def _predict_from_distances(
        self,
        distances: np.ndarray,
        fidelities: np.ndarray,
        y_train_fold: np.ndarray,
        k: int,
        weighted: bool,
        weighting_scheme: str,
    ) -> np.ndarray:
        """Internal helper for neighborhood voting."""
        n_queries = len(distances)
        predictions = np.zeros(n_queries, dtype=int)

        for i in range(n_queries):
            dists = distances[i]
            fids = fidelities[i]
            k_indices = np.argsort(dists)[:k]
            k_labels = y_train_fold[k_indices]

            if not weighted:
                # Standard unweighted majority voting
                unique_labels, counts = np.unique(k_labels, return_counts=True)
                winner = unique_labels[np.argmax(counts)]
                predictions[i] = winner
            else:
                # Weighted voting
                if weighting_scheme == "fidelity":
                    weights = fids[k_indices]
                else:  # inverse distance
                    weights = 1.0 / (dists[k_indices] + 1e-6)

                # Normalize weights
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    weights = weights / weight_sum

                class_weights: Dict[int, float] = {}
                for lbl, w in zip(k_labels, weights):
                    class_weights[lbl] = class_weights.get(lbl, 0.0) + float(w)

                winner = max(class_weights.keys(), key=lambda c: class_weights[c])
                predictions[i] = winner

        return predictions

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict test instances using the optimal feature map, optimal K, and weighted voting."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        start_time = time.perf_counter()
        X_test = np.asarray(X_test, dtype=np.float64)

        fidelities, distances = self.engine_.compute_fidelity_matrix(
            X_test, self.X_train_, states_train=self.states_train_
        )

        predictions = self._predict_from_distances(
            distances,
            fidelities,
            self.y_train_,
            self.best_k,
            self.weighted,
            self.weighting_scheme,
        )

        self.inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        return predictions

    def evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_labels: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Full evaluation on held-out test data."""
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
            "model_type": "Adaptive_Hybrid_QKNN (AH-QKNN)",
            "best_feature_map": self.best_feature_map,
            "best_k": self.best_k,
            "weighted": self.weighted,
            "weighting_scheme": self.weighting_scheme,
            "n_wires": self.n_wires,
            "depth": self.depth,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "class_labels": labels,
            "classification_report": report,
            "cv_results": self.cv_results_,
            "tuning_log": self.tuning_log_,
            "training_time_ms": round(self.training_time_ms, 2),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        }


def run_full_benchmark_comparison(
    dataset_data: Dict[str, Any],
    k: int = 3,
    candidate_k: Optional[List[int]] = None,
    n_wires: int = 4,
    depth: int = 1,
) -> Dict[str, Any]:
    """
    Executes all 6 models on identical dataset splits:
    1. Classical KNN (Baseline)
    2. Z-QKNN (Paper Baseline)
    3. ZZ-QKNN (Paper Baseline)
    4. Pauli-QKNN (Paper Baseline)
    5. Hybrid-QKNN (Paper Baseline)
    6. Adaptive Hybrid QKNN (AH-QKNN - Proposed Enhanced Model)

    Provides honest, empirical comparison without fabricated claims.
    """
    X_train = dataset_data["X_train_angles"]
    X_test = dataset_data["X_test_angles"]
    y_train = dataset_data["y_train"]
    y_test = dataset_data["y_test"]
    class_labels = dataset_data.get("class_labels")

    results = {}

    # 1. Classical KNN (uses standard scaled features)
    classical_model = ClassicalKNNBaseline(n_neighbors=k, metric="euclidean")
    results["classical_knn"] = classical_model.evaluate(
        dataset_data["X_train"], y_train, dataset_data["X_test"], y_test, class_labels=class_labels
    )

    # 2-5. Original QKNN Models (Z, ZZ, Pauli, Hybrid)
    for fmap in ["z", "zz", "pauli", "hybrid"]:
        qknn = QuantumKNNClassifier(
            feature_map=fmap, n_neighbors=k, n_wires=n_wires, depth=depth
        )
        results[f"{fmap}_qknn"] = qknn.evaluate(
            X_train, y_train, X_test, y_test, class_labels=class_labels
        )

    # 6. Proposed Adaptive Hybrid QKNN
    ah_qknn = AdaptiveHybridQKNN(
        candidate_feature_maps=["z", "zz", "pauli", "hybrid"],
        candidate_k=candidate_k or [1, 3, 5, 7],
        weighted=True,
        n_wires=n_wires,
        depth=depth,
        cv_folds=3,
        random_state=42,
    )
    results["adaptive_hybrid_qknn"] = ah_qknn.evaluate(
        X_train, y_train, X_test, y_test, class_labels=class_labels
    )

    # Calculate empirical delta
    hybrid_f1 = results["hybrid_qknn"]["f1_score"]
    ah_f1 = results["adaptive_hybrid_qknn"]["f1_score"]
    f1_delta = round(ah_f1 - hybrid_f1, 4)

    results["summary_analysis"] = {
        "hybrid_qknn_f1": hybrid_f1,
        "ah_qknn_f1": ah_f1,
        "f1_improvement": f1_delta,
        "improvement_detected": bool(f1_delta > 0),
        "selected_optimal_map": results["adaptive_hybrid_qknn"]["best_feature_map"],
        "selected_optimal_k": results["adaptive_hybrid_qknn"]["best_k"],
    }

    return results

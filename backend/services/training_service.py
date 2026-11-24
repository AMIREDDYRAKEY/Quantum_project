"""
Training, Evaluation, and Comparison Service
Executes side-by-side PennyLane QKNN and Scikit-Learn Classical KNN benchmarking.
"""
from typing import Dict, Any, List, Optional
import time
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
from backend.services.dataset_service import DatasetService
from backend.quantum.qknn_engine import QuantumSimilarityEngine


class TrainingService:
    @staticmethod
    def train_and_evaluate(
        dataset_id: str,
        k: int = 3,
        feature_map: str = "hybrid",
        weighted_voting: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes genuine side-by-side benchmark:
        1. Classical KNN (Scikit-Learn baseline)
        2. Quantum KNN (PennyLane default.qubit simulator)
        """
        session = DatasetService.get_session(dataset_id)
        if not session.get("configured") or not session.get("preprocessed_data"):
            raise ValueError(f"Dataset '{dataset_id}' must be configured before training.")

        data = session["preprocessed_data"]
        classes = data["classes"]
        n_classes = len(classes)
        k_val = min(k, len(data["y_train"]))

        # ==========================================
        # 1. Classical KNN (Baseline)
        # ==========================================
        t0_c_train = time.time()
        c_model = KNeighborsClassifier(n_neighbors=k_val, metric="euclidean")
        c_model.fit(data["X_train_reduced"], data["y_train"])
        t1_c_train = time.time()
        c_train_time = round(t1_c_train - t0_c_train, 4)

        t0_c_pred = time.time()
        c_preds = c_model.predict(data["X_test_reduced"])
        t1_c_pred = time.time()
        c_pred_time = round(t1_c_pred - t0_c_pred, 4)

        c_metrics = {
            "model_name": "Classical KNN (Scikit-Learn)",
            "accuracy": round(float(accuracy_score(data["y_test"], c_preds)), 4),
            "precision_macro": round(float(precision_score(data["y_test"], c_preds, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(data["y_test"], c_preds, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(data["y_test"], c_preds, average="macro", zero_division=0)), 4),
            "precision_weighted": round(float(precision_score(data["y_test"], c_preds, average="weighted", zero_division=0)), 4),
            "recall_weighted": round(float(recall_score(data["y_test"], c_preds, average="weighted", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(data["y_test"], c_preds, average="weighted", zero_division=0)), 4),
            "train_time_sec": c_train_time,
            "inference_time_sec": c_pred_time,
            "confusion_matrix": confusion_matrix(data["y_test"], c_preds).tolist(),
            "classification_report": classification_report(data["y_test"], c_preds, target_names=classes, output_dict=True, zero_division=0),
        }

        # ==========================================
        # 2. Quantum KNN (PennyLane Simulator)
        # ==========================================
        q_engine = QuantumSimilarityEngine(
            feature_map=feature_map,
            n_wires=data["n_qubits"],
            depth=1,
        )

        t0_q_train = time.time()
        # Precompute training quantum states
        train_states = np.array([
            q_engine.get_quantum_state(data["X_train_angles"][i])
            for i in range(len(data["X_train_angles"]))
        ])
        t1_q_train = time.time()
        q_train_time = round(t1_q_train - t0_q_train, 4)

        t0_q_pred = time.time()
        # Compute quantum state overlap fidelity matrix
        fidelities, distances = q_engine.compute_fidelity_matrix(
            data["X_test_angles"],
            data["X_train_angles"],
            states_train=train_states,
        )

        # K-Nearest neighbor selection & voting
        q_preds = []
        for i in range(len(data["X_test_angles"])):
            dist_row = distances[i]
            nearest_indices = np.argsort(dist_row)[:k_val]
            neighbor_labels = data["y_train"][nearest_indices]

            if weighted_voting:
                neighbor_fids = fidelities[i][nearest_indices]
                total_fid = np.sum(neighbor_fids)
                weights = neighbor_fids / total_fid if total_fid > 0 else np.ones(k_val) / k_val
                class_votes = {}
                for idx, lbl in enumerate(neighbor_labels):
                    class_votes[lbl] = class_votes.get(lbl, 0.0) + weights[idx]
                pred_label = max(class_votes.keys(), key=lambda c: class_votes[c])
            else:
                counts = np.bincount(neighbor_labels, minlength=n_classes)
                pred_label = int(np.argmax(counts))

            q_preds.append(pred_label)

        t1_q_pred = time.time()
        q_pred_time = round(t1_q_pred - t0_q_pred, 4)

        q_metrics = {
            "model_name": f"Quantum KNN ({feature_map.upper()} Map)",
            "accuracy": round(float(accuracy_score(data["y_test"], q_preds)), 4),
            "precision_macro": round(float(precision_score(data["y_test"], q_preds, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(data["y_test"], q_preds, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(data["y_test"], q_preds, average="macro", zero_division=0)), 4),
            "precision_weighted": round(float(precision_score(data["y_test"], q_preds, average="weighted", zero_division=0)), 4),
            "recall_weighted": round(float(recall_score(data["y_test"], q_preds, average="weighted", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(data["y_test"], q_preds, average="weighted", zero_division=0)), 4),
            "train_time_sec": q_train_time,
            "inference_time_sec": q_pred_time,
            "confusion_matrix": confusion_matrix(data["y_test"], q_preds).tolist(),
            "classification_report": classification_report(data["y_test"], q_preds, target_names=classes, output_dict=True, zero_division=0),
        }

        # Circuit metadata & ASCII
        circuit_meta = q_engine.get_circuit_metadata()
        circuit_ascii = q_engine.get_circuit_ascii(data["X_train_angles"][0])

        comparison = {
            "accuracy_delta": round(q_metrics["accuracy"] - c_metrics["accuracy"], 4),
            "f1_delta": round(q_metrics["f1_macro"] - c_metrics["f1_macro"], 4),
            "latency_ratio": round(q_pred_time / max(c_pred_time, 0.0001), 2),
            "higher_accuracy_model": "Quantum KNN" if q_metrics["accuracy"] > c_metrics["accuracy"] else ("Classical KNN" if c_metrics["accuracy"] > q_metrics["accuracy"] else "Tie"),
        }

        hardware_disclosure = (
            "Classical KNN runs on CPU vectorized arithmetic. "
            f"Quantum KNN was simulated using PennyLane's 'default.qubit' statevector simulator with {data['n_qubits']} qubits. "
            "Simulator execution on classical hardware incurs classical linear algebra simulation overhead ($O(2^N)$), "
            "whereas physical quantum hardware (QPU) executes state preparation and overlap fidelity directly via quantum measurement."
        )

        results_payload = {
            "dataset_id": dataset_id,
            "target_column": data["target_column"],
            "classes": classes,
            "k": k_val,
            "feature_map": feature_map,
            "qubits_used": data["n_qubits"],
            "selected_features": data["selected_features"],
            "classical_knn": c_metrics,
            "quantum_knn": q_metrics,
            "performance_comparison": comparison,
            "circuit_summary": {
                **circuit_meta,
                "ascii_diagram": circuit_ascii,
            },
            "hardware_disclosure": hardware_disclosure,
        }

        # Cache trained models in session
        session["trained"] = True
        session["trained_models"] = {
            "k": k_val,
            "feature_map": feature_map,
            "weighted_voting": weighted_voting,
            "classical_model": c_model,
            "quantum_engine": q_engine,
            "train_states": train_states,
            "results": results_payload,
        }

        return results_payload

    @staticmethod
    def predict_single(dataset_id: str, input_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Executes live inference on user input using both Classical KNN and Quantum KNN.
        """
        session = DatasetService.get_session(dataset_id)
        if not session.get("trained") or not session.get("trained_models"):
            raise ValueError("Model must be trained on this dataset before making predictions.")

        data = session["preprocessed_data"]
        trained = session["trained_models"]
        classes = data["classes"]
        k_val = trained["k"]

        # Parse and order features according to original feature schema
        orig_cols = data["original_features"]
        raw_row = []
        for col in orig_cols:
            val = input_features.get(col, 0.0)
            raw_row.append(float(val))

        input_arr = np.array(raw_row, dtype=np.float64).reshape(1, -1)

        # Transform using the trained scaler
        scaled_row = data["scaler"].transform(input_arr)

        # Apply feature selection/PCA if used during training
        if data["selector"] is not None:
            reduced_row = data["selector"].transform(scaled_row)
        else:
            reduced_row = scaled_row

        # Angle scale: 2 * arctan(x)
        angles_row = 2.0 * np.arctan(reduced_row)

        # 1. Classical Prediction
        c_model = trained["classical_model"]
        c_pred_idx = int(c_model.predict(reduced_row)[0])
        c_pred_label = classes[c_pred_idx]
        c_dists, c_indices = c_model.kneighbors(reduced_row)

        # 2. Quantum Prediction
        q_engine = trained["quantum_engine"]
        fids, q_dists = q_engine.compute_fidelity_matrix(
            angles_row,
            data["X_train_angles"],
            states_train=trained["train_states"],
        )

        nearest_q_indices = np.argsort(q_dists[0])[:k_val]
        neighbor_labels = data["y_train"][nearest_q_indices]
        neighbor_fids = fids[0][nearest_q_indices]
        total_fid = np.sum(neighbor_fids)

        weights = neighbor_fids / total_fid if total_fid > 0 else np.ones(k_val) / k_val
        class_votes = {}
        for idx, lbl in enumerate(neighbor_labels):
            class_votes[lbl] = class_votes.get(lbl, 0.0) + weights[idx]

        q_pred_idx = int(max(class_votes.keys(), key=lambda c: class_votes[c]))
        q_pred_label = classes[q_pred_idx]
        q_confidence = round(float(class_votes[q_pred_idx]), 4)

        # Format Nearest Neighbors Table
        neighbors_info = []
        for rank in range(k_val):
            idx = int(nearest_q_indices[rank])
            lbl_idx = int(data["y_train"][idx])
            neighbors_info.append({
                "rank": rank + 1,
                "train_index": idx,
                "class_label": classes[lbl_idx],
                "fidelity": round(float(fids[0][idx]), 4),
                "quantum_distance": round(float(q_dists[0][idx]), 4),
                "classical_distance": round(float(c_dists[0][rank]), 4) if rank < len(c_dists[0]) else None,
                "voting_weight": round(float(weights[rank]), 4),
            })

        agreement = (q_pred_idx == c_pred_idx)
        explanation = (
            f"Quantum KNN predicted '{q_pred_label}' with {q_confidence*100:.1f}% weighted fidelity vote. "
            f"Classical KNN predicted '{c_pred_label}' via Euclidean distance. "
            f"{'Both models agree on classification.' if agreement else 'Models differ due to quantum feature space geometry.'}"
        )

        return {
            "quantum_predicted_class": q_pred_label,
            "classical_predicted_class": c_pred_label,
            "agreement": agreement,
            "quantum_confidence": q_confidence,
            "classical_confidence": round(float(np.mean(c_model.predict_proba(reduced_row)[0])), 4) if hasattr(c_model, "predict_proba") else 1.0,
            "nearest_neighbors": neighbors_info,
            "explanation": explanation,
        }

    @staticmethod
    def predict_batch(dataset_id: str, records: List[Dict[str, float]]) -> Dict[str, Any]:
        """Runs batch predictions across multiple input rows."""
        predictions = []
        for i, row in enumerate(records):
            res = TrainingService.predict_single(dataset_id, row)
            predictions.append({
                "row_index": i + 1,
                "quantum_predicted_class": res["quantum_predicted_class"],
                "classical_predicted_class": res["classical_predicted_class"],
                "agreement": res["agreement"],
                "quantum_confidence": res["quantum_confidence"],
            })
        return {
            "total_records": len(records),
            "predictions": predictions,
        }

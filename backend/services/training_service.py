"""
Training, Evaluation, and Comparison Service
Executes comprehensive side-by-side benchmarking:
1. Classical KNN (Scikit-Learn baseline)
2. Z Feature Map QKNN (PennyLane default.qubit)
3. ZZ Feature Map QKNN (PennyLane default.qubit)
4. Pauli Feature Map QKNN (PennyLane default.qubit)
5. Hybrid Feature Map QKNN (PennyLane default.qubit)
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
        Executes genuine side-by-side benchmark comparing:
        1. Classical KNN (Scikit-Learn baseline)
        2. All four PennyLane Quantum Feature Maps: Z, ZZ, Pauli, and Hybrid.
        Provides honest, empirical performance comparisons without fabricated metrics.
        """
        session = DatasetService.get_session(dataset_id)
        if not session.get("configured") or not session.get("preprocessed_data"):
            raise ValueError(f"Dataset '{dataset_id}' must be configured before training.")

        data = session["preprocessed_data"]
        classes = data["classes"]
        n_classes = len(classes)
        k_val = max(1, min(k, len(data["y_train"])))

        # ==========================================
        # 1. Classical KNN Baseline (Scikit-Learn)
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
            "model_key": "classical",
            "accuracy": round(float(accuracy_score(data["y_test"], c_preds)), 4),
            "precision_macro": round(float(precision_score(data["y_test"], c_preds, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(data["y_test"], c_preds, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(data["y_test"], c_preds, average="macro", zero_division=0)), 4),
            "precision_weighted": round(float(precision_score(data["y_test"], c_preds, average="weighted", zero_division=0)), 4),
            "recall_weighted": round(float(recall_score(data["y_test"], c_preds, average="weighted", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(data["y_test"], c_preds, average="weighted", zero_division=0)), 4),
            "train_time_sec": c_train_time,
            "inference_time_sec": c_pred_time,
            "confusion_matrix": confusion_matrix(data["y_test"], c_preds, labels=list(range(n_classes))).tolist(),
            "classification_report": classification_report(data["y_test"], c_preds, target_names=classes, output_dict=True, zero_division=0),
        }

        # ==========================================
        # 2. Evaluate All Quantum Feature Maps (PennyLane default.qubit)
        # ==========================================
        # Always evaluate the 4 core feature maps (Z, ZZ, Pauli, Hybrid)
        maps_to_evaluate = ["z", "zz", "pauli", "hybrid"]
        primary_map = feature_map.lower().strip()
        if primary_map not in maps_to_evaluate:
            maps_to_evaluate.append(primary_map)

        q_models: Dict[str, Dict[str, Any]] = {}
        q_engines: Dict[str, QuantumSimilarityEngine] = {}
        train_states_dict: Dict[str, np.ndarray] = {}
        all_circuits: Dict[str, Any] = {}

        for fmap in maps_to_evaluate:
            q_engine = QuantumSimilarityEngine(
                feature_map=fmap,
                n_wires=data["n_qubits"],
                depth=1,
            )
            q_engines[fmap] = q_engine

            # Circuit metadata & schematic
            circuit_meta = q_engine.get_circuit_metadata()
            circuit_ascii = q_engine.get_circuit_ascii(data["X_train_angles"][0])
            all_circuits[fmap] = {
                **circuit_meta,
                "ascii_diagram": circuit_ascii,
            }

            # Precompute training quantum states
            t0_q_train = time.time()
            t_states = np.array([
                q_engine.get_quantum_state(data["X_train_angles"][i])
                for i in range(len(data["X_train_angles"]))
            ])
            t1_q_train = time.time()
            train_states_dict[fmap] = t_states
            q_train_time = round(t1_q_train - t0_q_train, 4)

            # Compute test states and quantum state overlap fidelity matrix
            t0_q_pred = time.time()
            fidelities, distances = q_engine.compute_fidelity_matrix(
                data["X_test_angles"],
                data["X_train_angles"],
                states_train=t_states,
            )

            # K-Nearest neighbor classification via fidelity-weighted voting
            q_preds = []
            for i in range(len(data["X_test_angles"])):
                dist_row = distances[i]
                nearest_indices = np.argsort(dist_row)[:k_val]
                neighbor_labels = data["y_train"][nearest_indices]

                if weighted_voting:
                    neighbor_fids = fidelities[i][nearest_indices]
                    total_fid = np.sum(neighbor_fids)
                    weights = neighbor_fids / total_fid if total_fid > 0 else np.ones(k_val) / k_val
                    class_votes: Dict[int, float] = {}
                    for idx, lbl in enumerate(neighbor_labels):
                        class_votes[lbl] = class_votes.get(lbl, 0.0) + weights[idx]
                    # Deterministic tie-breaking
                    pred_label = max(sorted(class_votes.keys()), key=lambda c: class_votes[c])
                else:
                    counts = np.bincount(neighbor_labels, minlength=n_classes)
                    pred_label = int(np.argmax(counts))

                q_preds.append(pred_label)

            t1_q_pred = time.time()
            q_pred_time = round(t1_q_pred - t0_q_pred, 4)

            q_models[fmap] = {
                "model_name": f"{fmap.upper()} Feature Map QKNN",
                "model_key": fmap,
                "accuracy": round(float(accuracy_score(data["y_test"], q_preds)), 4),
                "precision_macro": round(float(precision_score(data["y_test"], q_preds, average="macro", zero_division=0)), 4),
                "recall_macro": round(float(recall_score(data["y_test"], q_preds, average="macro", zero_division=0)), 4),
                "f1_macro": round(float(f1_score(data["y_test"], q_preds, average="macro", zero_division=0)), 4),
                "precision_weighted": round(float(precision_score(data["y_test"], q_preds, average="weighted", zero_division=0)), 4),
                "recall_weighted": round(float(recall_score(data["y_test"], q_preds, average="weighted", zero_division=0)), 4),
                "f1_weighted": round(float(f1_score(data["y_test"], q_preds, average="weighted", zero_division=0)), 4),
                "train_time_sec": q_train_time,
                "inference_time_sec": q_pred_time,
                "total_time_sec": round(q_train_time + q_pred_time, 4),
                "confusion_matrix": confusion_matrix(data["y_test"], q_preds, labels=list(range(n_classes))).tolist(),
                "classification_report": classification_report(data["y_test"], q_preds, target_names=classes, output_dict=True, zero_division=0),
                "circuit_summary": all_circuits[fmap],
            }

        # Primary selected quantum model (defaults to hybrid)
        primary_q_metrics = q_models.get(primary_map, q_models["hybrid"])

        # ==========================================
        # 3. Construct 5-Model Benchmark Comparison Table
        # ==========================================
        comparison_table = [
            {
                "model_key": "classical",
                "model_name": "Classical KNN (Scikit-Learn)",
                "display_name": "Classical KNN (Scikit-Learn)",
                "feature_map": "classical",
                "paradigm": "Classical",
                "accuracy": c_metrics["accuracy"],
                "precision": c_metrics["precision_macro"],
                "precision_macro": c_metrics["precision_macro"],
                "recall": c_metrics["recall_macro"],
                "recall_macro": c_metrics["recall_macro"],
                "f1_score": c_metrics["f1_macro"],
                "f1_macro": c_metrics["f1_macro"],
                "train_time_sec": c_metrics["train_time_sec"],
                "inference_time_sec": c_metrics["inference_time_sec"],
                "latency_sec": c_metrics["inference_time_sec"],
                "total_time_sec": round(c_metrics["train_time_sec"] + c_metrics["inference_time_sec"], 4),
                "is_quantum": False,
                "is_best_model": False,
                "is_best_quantum": False,
            },
            {
                "model_key": "z",
                "model_name": "Z Feature Map QKNN",
                "display_name": "Z Feature Map QKNN",
                "feature_map": "z",
                "paradigm": "Quantum",
                "accuracy": q_models["z"]["accuracy"],
                "precision": q_models["z"]["precision_macro"],
                "precision_macro": q_models["z"]["precision_macro"],
                "recall": q_models["z"]["recall_macro"],
                "recall_macro": q_models["z"]["recall_macro"],
                "f1_score": q_models["z"]["f1_macro"],
                "f1_macro": q_models["z"]["f1_macro"],
                "train_time_sec": q_models["z"]["train_time_sec"],
                "inference_time_sec": q_models["z"]["inference_time_sec"],
                "latency_sec": q_models["z"]["inference_time_sec"],
                "total_time_sec": q_models["z"]["total_time_sec"],
                "is_quantum": True,
                "is_best_model": False,
                "is_best_quantum": False,
            },
            {
                "model_key": "zz",
                "model_name": "ZZ Feature Map QKNN",
                "display_name": "ZZ Feature Map QKNN",
                "feature_map": "zz",
                "paradigm": "Quantum",
                "accuracy": q_models["zz"]["accuracy"],
                "precision": q_models["zz"]["precision_macro"],
                "precision_macro": q_models["zz"]["precision_macro"],
                "recall": q_models["zz"]["recall_macro"],
                "recall_macro": q_models["zz"]["recall_macro"],
                "f1_score": q_models["zz"]["f1_macro"],
                "f1_macro": q_models["zz"]["f1_macro"],
                "train_time_sec": q_models["zz"]["train_time_sec"],
                "inference_time_sec": q_models["zz"]["inference_time_sec"],
                "latency_sec": q_models["zz"]["inference_time_sec"],
                "total_time_sec": q_models["zz"]["total_time_sec"],
                "is_quantum": True,
                "is_best_model": False,
                "is_best_quantum": False,
            },
            {
                "model_key": "pauli",
                "model_name": "Pauli Feature Map QKNN",
                "display_name": "Pauli Feature Map QKNN",
                "feature_map": "pauli",
                "paradigm": "Quantum",
                "accuracy": q_models["pauli"]["accuracy"],
                "precision": q_models["pauli"]["precision_macro"],
                "precision_macro": q_models["pauli"]["precision_macro"],
                "recall": q_models["pauli"]["recall_macro"],
                "recall_macro": q_models["pauli"]["recall_macro"],
                "f1_score": q_models["pauli"]["f1_macro"],
                "f1_macro": q_models["pauli"]["f1_macro"],
                "train_time_sec": q_models["pauli"]["train_time_sec"],
                "inference_time_sec": q_models["pauli"]["inference_time_sec"],
                "latency_sec": q_models["pauli"]["inference_time_sec"],
                "total_time_sec": q_models["pauli"]["total_time_sec"],
                "is_quantum": True,
                "is_best_model": False,
                "is_best_quantum": False,
            },
            {
                "model_key": "hybrid",
                "model_name": "Hybrid Feature Map QKNN",
                "display_name": "Hybrid Feature Map QKNN",
                "feature_map": "hybrid",
                "paradigm": "Quantum",
                "accuracy": q_models["hybrid"]["accuracy"],
                "precision": q_models["hybrid"]["precision_macro"],
                "precision_macro": q_models["hybrid"]["precision_macro"],
                "recall": q_models["hybrid"]["recall_macro"],
                "recall_macro": q_models["hybrid"]["recall_macro"],
                "f1_score": q_models["hybrid"]["f1_macro"],
                "f1_macro": q_models["hybrid"]["f1_macro"],
                "train_time_sec": q_models["hybrid"]["train_time_sec"],
                "inference_time_sec": q_models["hybrid"]["inference_time_sec"],
                "latency_sec": q_models["hybrid"]["inference_time_sec"],
                "total_time_sec": q_models["hybrid"]["total_time_sec"],
                "is_quantum": True,
                "is_best_model": False,
                "is_best_quantum": False,
            },
        ]

        # Identify best overall model and best quantum model
        best_overall_entry = max(comparison_table, key=lambda m: (m["accuracy"], m["f1_score"]))
        best_overall_model = best_overall_entry["model_name"]

        quantum_entries = [m for m in comparison_table if m["is_quantum"]]
        best_quantum_entry = max(quantum_entries, key=lambda m: (m["accuracy"], m["f1_score"]))
        best_quantum_map = best_quantum_entry["model_name"]

        # Tag leading flags
        for m in comparison_table:
            if m["model_name"] == best_overall_model or m["model_key"] == best_overall_model:
                m["is_best_model"] = True
            if m["model_name"] == best_quantum_map or m["model_key"] == best_quantum_map:
                m["is_best_quantum"] = True

        # Empirical Hybrid vs Individual Feature Map Analysis
        hybrid_m = q_models["hybrid"]
        delta_vs_z = round(hybrid_m["accuracy"] - q_models["z"]["accuracy"], 4)
        delta_f1_vs_z = round(hybrid_m["f1_macro"] - q_models["z"]["f1_macro"], 4)

        delta_vs_zz = round(hybrid_m["accuracy"] - q_models["zz"]["accuracy"], 4)
        delta_f1_vs_zz = round(hybrid_m["f1_macro"] - q_models["zz"]["f1_macro"], 4)

        delta_vs_pauli = round(hybrid_m["accuracy"] - q_models["pauli"]["accuracy"], 4)
        delta_f1_vs_pauli = round(hybrid_m["f1_macro"] - q_models["pauli"]["f1_macro"], 4)

        delta_vs_classical = round(hybrid_m["accuracy"] - c_metrics["accuracy"], 4)
        delta_f1_vs_classical = round(hybrid_m["f1_macro"] - c_metrics["f1_macro"], 4)

        if delta_vs_z >= 0 and delta_vs_zz >= 0 and delta_vs_pauli >= 0:
            hybrid_insight = (
                f"Hybrid Feature Map demonstrated superior or equal state space separability "
                f"compared to individual maps (+{delta_vs_z*100:.1f}% vs Z, +{delta_vs_zz*100:.1f}% vs ZZ, +{delta_vs_pauli*100:.1f}% vs Pauli)."
            )
        else:
            hybrid_insight = (
                f"Hybrid Feature Map attained {hybrid_m['accuracy']*100:.1f}% accuracy "
                f"({delta_vs_z*100:+.1f}% vs Z, {delta_vs_zz*100:+.1f}% vs ZZ, {delta_vs_pauli*100:+.1f}% vs Pauli). "
                f"Individual feature maps produced sharper or competitive localized margins for this dataset."
            )

        comparison = {
            "accuracy_delta": round(primary_q_metrics["accuracy"] - c_metrics["accuracy"], 4),
            "f1_delta": round(primary_q_metrics["f1_macro"] - c_metrics["f1_macro"], 4),
            "latency_ratio": round(primary_q_metrics["inference_time_sec"] / max(c_metrics["inference_time_sec"], 0.0001), 2),
            "higher_accuracy_model": "Quantum KNN" if primary_q_metrics["accuracy"] > c_metrics["accuracy"] else ("Classical KNN" if c_metrics["accuracy"] > primary_q_metrics["accuracy"] else "Tie"),
            "best_model": best_overall_model,
            "best_quantum_map": best_quantum_map,
        }

        hybrid_analysis = {
            "hybrid_accuracy": hybrid_m["accuracy"],
            "hybrid_f1": hybrid_m["f1_macro"],
            "delta_vs_z_accuracy": delta_vs_z,
            "delta_accuracy_vs_z": delta_vs_z,
            "delta_f1_vs_z": delta_f1_vs_z,
            "delta_vs_zz_accuracy": delta_vs_zz,
            "delta_accuracy_vs_zz": delta_vs_zz,
            "delta_f1_vs_zz": delta_f1_vs_zz,
            "delta_vs_pauli_accuracy": delta_vs_pauli,
            "delta_accuracy_vs_pauli": delta_vs_pauli,
            "delta_f1_vs_pauli": delta_f1_vs_pauli,
            "delta_vs_classical_accuracy": delta_vs_classical,
            "delta_accuracy_vs_classical": delta_vs_classical,
            "delta_f1_vs_classical": delta_f1_vs_classical,
            "best_feature_map": best_quantum_map,
            "insight": hybrid_insight,
            "recommendation": hybrid_insight,
        }

        hardware_disclosure = (
            "Classical KNN runs on CPU vectorized arithmetic. "
            f"Quantum KNN models were simulated using PennyLane's 'default.qubit' statevector simulator with {data['n_qubits']} qubits. "
            "Classical CPU simulation incurs matrix exponential overhead ($O(2^N)$), "
            "whereas physical quantum hardware (QPU) computes state preparation and overlap fidelity directly via quantum swap tests."
        )

        results_payload = {
            "dataset_id": dataset_id,
            "target_column": data["target_column"],
            "classes": classes,
            "k": k_val,
            "feature_map": primary_map,
            "qubits_used": data["n_qubits"],
            "selected_features": data["selected_features"],
            "classical_knn": c_metrics,
            "quantum_knn": primary_q_metrics,
            "all_quantum_models": q_models,
            "comparison_table": comparison_table,
            "best_model": best_overall_model,
            "best_quantum_map": best_quantum_map,
            "hybrid_analysis": hybrid_analysis,
            "performance_comparison": comparison,
            "circuit_summary": all_circuits.get(primary_map, all_circuits["hybrid"]),
            "all_circuits": all_circuits,
            "hardware_disclosure": hardware_disclosure,
        }

        # Cache trained models and quantum engines in session
        session["trained"] = True
        session["trained_models"] = {
            "k": k_val,
            "feature_map": primary_map,
            "weighted_voting": weighted_voting,
            "classical_model": c_model,
            "quantum_engine": q_engines.get(primary_map, q_engines["hybrid"]),
            "train_states": train_states_dict.get(primary_map, train_states_dict["hybrid"]),
            "quantum_engines": q_engines,
            "train_states_dict": train_states_dict,
            "results": results_payload,
        }

        # Automatically generate and persist official PDF research report immediately
        try:
            from backend.reports.pdf_generator import ReportGenerator
            pdf_path = ReportGenerator.generate_pdf_report(dataset_id)
            session["report_path"] = str(pdf_path)
            results_payload["report_path"] = str(pdf_path)
            results_payload["report_ready"] = True
        except Exception as report_err:
            session["report_path"] = None
            results_payload["report_ready"] = False
            results_payload["report_generation_warning"] = str(report_err)

        return results_payload

    @staticmethod
    def predict_single(
        dataset_id: str,
        input_features: Dict[str, float],
        model_type: str = "hybrid",
    ) -> Dict[str, Any]:
        """
        Executes live inference on user input for any chosen model:
        'classical', 'z', 'zz', 'pauli', or 'hybrid'.
        Does not retrain models. Uses cached state representations and scalers.
        """
        session = DatasetService.get_session(dataset_id)
        if not session.get("trained") or not session.get("trained_models"):
            raise ValueError("Model must be trained on this dataset before making predictions.")

        data = session["preprocessed_data"]
        trained = session["trained_models"]
        classes = data["classes"]
        k_val = trained["k"]
        req_model = model_type.lower().strip()

        # Parse and order features according to original feature schema
        orig_cols = data["original_features"]
        feature_medians = data.get("feature_medians", {})
        raw_row = []
        for col in orig_cols:
            val = input_features.get(col)
            if val is not None:
                try:
                    raw_row.append(float(val))
                except (ValueError, TypeError):
                    raw_row.append(feature_medians.get(col, 0.0))
            else:
                raw_row.append(feature_medians.get(col, 0.0))

        input_arr = np.array(raw_row, dtype=np.float64).reshape(1, -1)

        # Transform using the trained scaler (zero leakage)
        scaled_row = data["scaler"].transform(input_arr)

        # Apply feature selection/PCA if used during training
        if data["selector"] is not None:
            reduced_row = data["selector"].transform(scaled_row)
        else:
            reduced_row = scaled_row

        # Angle scale: 2 * arctan(x)
        angles_row = 2.0 * np.arctan(reduced_row)

        # 1. Classical Prediction (always computed for baseline reference)
        c_model = trained["classical_model"]
        c_pred_idx = int(c_model.predict(reduced_row)[0])
        c_pred_label = classes[c_pred_idx]
        c_dists, c_indices = c_model.kneighbors(reduced_row)

        # 2. Select Quantum Engine for requested model
        q_engines = trained.get("quantum_engines", {})
        train_states_dict = trained.get("train_states_dict", {})

        selected_q_map = req_model if req_model in q_engines else trained["feature_map"]
        q_engine = q_engines.get(selected_q_map, trained["quantum_engine"])
        t_states = train_states_dict.get(selected_q_map, trained["train_states"])

        # Compute quantum state overlap fidelity against cached training states
        fids, q_dists = q_engine.compute_fidelity_matrix(
            angles_row,
            data["X_train_angles"],
            states_train=t_states,
        )

        nearest_q_indices = np.argsort(q_dists[0])[:k_val]
        neighbor_labels = data["y_train"][nearest_q_indices]
        neighbor_fids = fids[0][nearest_q_indices]
        total_fid = np.sum(neighbor_fids)

        weights = neighbor_fids / total_fid if total_fid > 0 else np.ones(k_val) / k_val
        class_votes: Dict[int, float] = {}
        for idx, lbl in enumerate(neighbor_labels):
            class_votes[lbl] = class_votes.get(lbl, 0.0) + weights[idx]

        q_pred_idx = int(max(sorted(class_votes.keys()), key=lambda c: class_votes[c]))
        q_pred_label = classes[q_pred_idx]
        q_confidence = round(float(class_votes[q_pred_idx]), 4)

        # If user explicitly requested Classical KNN as the primary prediction
        if req_model == "classical":
            primary_pred = c_pred_label
            model_display = "Classical KNN (Scikit-Learn)"
            primary_conf = 1.0
        else:
            primary_pred = q_pred_label
            model_display = f"Quantum KNN ({selected_q_map.upper()} Feature Map)"
            primary_conf = q_confidence

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
            f"{model_display} predicted '{primary_pred}' "
            f"(Confidence: {primary_conf*100:.1f}%). "
            f"Classical KNN predicted '{c_pred_label}' via Euclidean metric. "
            f"{'Both quantum and classical representations agree.' if agreement else 'Quantum Hilbert state space geometry separated this instance differently from classical Euclidean space.'}"
        )

        return {
            "model_used": model_display,
            "selected_feature_map": selected_q_map,
            "predicted_class": primary_pred,
            "confidence": primary_conf,
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

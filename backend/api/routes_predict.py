"""Real-Time Inference Playground API Endpoints"""
from flask import Blueprint, jsonify, request
import numpy as np
from backend.core.data_manager import DataManager
from backend.core.quantum_knn import QuantumKNNClassifier
from backend.core.classical_knn import ClassicalKNNBaseline
from backend.core.adaptive_hybrid_qknn import AdaptiveHybridQKNN

predict_bp = Blueprint("predict", __name__, url_prefix="/api/predict")
dm = DataManager(random_state=42)


@predict_bp.route("/single", methods=["POST"])
def predict_single():
    """
    Real-time inference endpoint for custom feature input.
    Accepts: { dataset, model_type, features: [x1, x2, ...] }
    """
    try:
        body = request.get_json() or {}
        dataset_name = body.get("dataset", "iris")
        model_type = body.get("model_type", "adaptive_hybrid_qknn").lower()
        feature_map = body.get("feature_map", "hybrid").lower()
        k = int(body.get("k", 3))
        raw_features = body.get("features", [])

        if not raw_features:
            return jsonify({"status": "error", "message": "Missing 'features' list"}), 400

        n_wires = 4
        data = dm.split_and_preprocess(
            dataset_name,
            n_features=n_wires if dataset_name != "iris" else None,
            angle_scale=True,
        )

        input_arr = np.array(raw_features, dtype=np.float64).reshape(1, -1)

        # Scale features using the trained scaler
        if data["scaler"] is not None:
            input_scaled = data["scaler"].transform(input_arr)
        else:
            input_scaled = input_arr

        # Apply feature selection if high dimensional
        if data["selector"] is not None:
            input_scaled = data["selector"].transform(input_scaled)

        # Angle scale for quantum models: 2 * arctan(x)
        input_angles = 2.0 * np.arctan(input_scaled)

        labels = data["class_labels"]

        if model_type == "classical_knn":
            model = ClassicalKNNBaseline(n_neighbors=k)
            model.fit(data["X_train"], data["y_train"])
            pred = int(model.predict(input_scaled)[0])
            neighbors_dist, neighbors_idx = model.kneighbors(input_scaled)
            neighbors = [
                {
                    "rank": i + 1,
                    "train_index": int(neighbors_idx[0][i]),
                    "label": int(data["y_train"][neighbors_idx[0][i]]),
                    "distance": round(float(neighbors_dist[0][i]), 4),
                }
                for i in range(len(neighbors_idx[0]))
            ]
            winning_label = labels[pred] if pred < len(labels) else str(pred)

            return jsonify({
                "status": "success",
                "model_type": "Classical_KNN",
                "predicted_class": pred,
                "predicted_label": winning_label,
                "neighbors": neighbors,
            })

        elif model_type in ["adaptive_hybrid_qknn", "ah_qknn"]:
            ah_model = AdaptiveHybridQKNN(
                candidate_feature_maps=["z", "zz", "pauli", "hybrid"],
                candidate_k=[1, 3, 5],
                weighted=True,
                n_wires=n_wires,
            )
            ah_model.fit(data["X_train_angles"], data["y_train"])
            pred = int(ah_model.predict(input_angles)[0])
            winning_label = labels[pred] if pred < len(labels) else str(pred)

            # Extract neighbor fidelities
            fids, dists = ah_model.engine_.compute_fidelity_matrix(
                input_angles, data["X_train_angles"], states_train=ah_model.states_train_
            )
            k_indices = np.argsort(dists[0])[: ah_model.best_k]
            neighbors = [
                {
                    "rank": i + 1,
                    "train_index": int(idx),
                    "label": int(data["y_train"][idx]),
                    "fidelity": round(float(fids[0][idx]), 4),
                    "quantum_distance": round(float(dists[0][idx]), 4),
                }
                for i, idx in enumerate(k_indices)
            ]

            return jsonify({
                "status": "success",
                "model_type": "Adaptive_Hybrid_QKNN",
                "best_feature_map": ah_model.best_feature_map,
                "best_k": ah_model.best_k,
                "predicted_class": pred,
                "predicted_label": winning_label,
                "neighbors": neighbors,
            })

        else:  # Standard Quantum KNN with selected map
            qknn = QuantumKNNClassifier(feature_map=feature_map, n_neighbors=k, n_wires=n_wires)
            qknn.fit(data["X_train_angles"], data["y_train"])
            pred = int(qknn.predict(input_angles)[0])
            winning_label = labels[pred] if pred < len(labels) else str(pred)
            details = qknn.get_neighbor_details(input_angles[0])

            return jsonify({
                "status": "success",
                "model_type": f"Quantum_KNN_{feature_map.upper()}",
                "feature_map": feature_map,
                "predicted_class": pred,
                "predicted_label": winning_label,
                "neighbors": details["neighbors"],
            })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

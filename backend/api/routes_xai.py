"""Explainable AI (XAI) API Endpoints"""
from flask import Blueprint, jsonify, request
from backend.core.data_manager import DataManager
from backend.core.classical_knn import ClassicalKNNBaseline
from backend.core.quantum_knn import QuantumKNNClassifier
from backend.core.explainability import ExplainabilityEngine

xai_bp = Blueprint("xai", __name__, url_prefix="/api/xai")
dm = DataManager(random_state=42)


@xai_bp.route("/classical", methods=["POST"])
def explain_classical():
    """Compute SHAP feature importances for classical baseline model."""
    try:
        body = request.get_json() or {}
        dataset_name = body.get("dataset", "iris")
        k = int(body.get("k", 3))
        sample_idx = int(body.get("sample_index", 0))

        data = dm.split_and_preprocess(dataset_name)
        model = ClassicalKNNBaseline(n_neighbors=k)
        model.fit(data["X_train"], data["y_train"])

        sample = data["X_test"][sample_idx % len(data["X_test"])]
        shap_explanation = ExplainabilityEngine.explain_classical_predictions(
            model,
            data["X_train"],
            sample,
            data["selected_features"],
            max_background=25,
        )

        return jsonify({
            "status": "success",
            "dataset": dataset_name,
            "explanation": shap_explanation,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@xai_bp.route("/quantum", methods=["POST"])
def explain_quantum():
    """Compute quantum feature sensitivities and neighbor contributions."""
    try:
        body = request.get_json() or {}
        dataset_name = body.get("dataset", "iris")
        feature_map = body.get("feature_map", "hybrid")
        k = int(body.get("k", 3))
        sample_idx = int(body.get("sample_index", 0))
        n_wires = int(body.get("n_wires", 4))

        data = dm.split_and_preprocess(
            dataset_name,
            n_features=n_wires if dataset_name != "iris" else None,
            angle_scale=True,
        )

        qknn = QuantumKNNClassifier(feature_map=feature_map, n_neighbors=k, n_wires=n_wires)
        qknn.fit(data["X_train_angles"], data["y_train"])

        sample = data["X_test_angles"][sample_idx % len(data["X_test_angles"])]
        explanation = ExplainabilityEngine.explain_quantum_prediction(
            qknn,
            sample,
            data["selected_features"],
            class_labels=data["class_labels"],
        )

        return jsonify({
            "status": "success",
            "dataset": dataset_name,
            "explanation": explanation,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@xai_bp.route("/feature-maps", methods=["GET"])
def get_feature_maps_telemetry():
    """Return comparative circuit telemetry for all 4 quantum feature maps."""
    try:
        n_qubits = int(request.args.get("n_qubits", 4))
        depth = int(request.args.get("depth", 1))
        telemetry = ExplainabilityEngine.get_feature_maps_comparison_telemetry(
            n_qubits=n_qubits, depth=depth
        )
        return jsonify({"status": "success", "feature_maps": telemetry})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

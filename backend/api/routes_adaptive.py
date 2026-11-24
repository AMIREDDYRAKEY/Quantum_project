"""Adaptive Hybrid Quantum KNN (AH-QKNN) API Endpoints"""
from flask import Blueprint, jsonify, request
from backend.core.data_manager import DataManager
from backend.core.adaptive_hybrid_qknn import AdaptiveHybridQKNN

adaptive_bp = Blueprint("adaptive", __name__, url_prefix="/api/adaptive")
dm = DataManager(random_state=42)


@adaptive_bp.route("/train", methods=["POST"])
def train_adaptive_hybrid():
    """Train and evaluate proposed Adaptive Hybrid QKNN with CV tuning."""
    try:
        body = request.get_json() or {}
        dataset_name = body.get("dataset", "iris")
        candidate_maps = body.get("candidate_feature_maps", ["z", "zz", "pauli", "hybrid"])
        candidate_k = body.get("candidate_k", [1, 3, 5])
        weighted = bool(body.get("weighted", True))
        weighting_scheme = body.get("weighting_scheme", "fidelity")
        n_wires = int(body.get("n_wires", 4))
        depth = int(body.get("depth", 1))
        cv_folds = int(body.get("cv_folds", 3))

        data = dm.split_and_preprocess(
            dataset_name,
            n_features=n_wires if dataset_name != "iris" else None,
            angle_scale=True,
        )

        model = AdaptiveHybridQKNN(
            candidate_feature_maps=candidate_maps,
            candidate_k=candidate_k,
            weighted=weighted,
            weighting_scheme=weighting_scheme,
            n_wires=n_wires,
            depth=depth,
            cv_folds=cv_folds,
            random_state=42,
        )

        eval_result = model.evaluate(
            data["X_train_angles"],
            data["y_train"],
            data["X_test_angles"],
            data["y_test"],
            class_labels=data["class_labels"],
        )

        return jsonify({
            "status": "success",
            "dataset": dataset_name,
            "results": eval_result,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

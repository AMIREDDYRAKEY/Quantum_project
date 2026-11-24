"""Classical KNN API Endpoints"""
from flask import Blueprint, jsonify, request
from backend.core.data_manager import DataManager
from backend.core.classical_knn import ClassicalKNNBaseline

classical_bp = Blueprint("classical", __name__, url_prefix="/api/classical")
dm = DataManager(random_state=42)


@classical_bp.route("/train", methods=["POST"])
def train_classical_knn():
    """Train and evaluate Classical KNN baseline."""
    try:
        body = request.get_json() or {}
        dataset_name = body.get("dataset", "iris")
        k = int(body.get("k", 3))
        metric = body.get("metric", "euclidean")
        weights = body.get("weights", "uniform")
        test_size = float(body.get("test_size", 0.20))

        data = dm.split_and_preprocess(dataset_name, test_size=test_size)
        model = ClassicalKNNBaseline(n_neighbors=k, metric=metric, weights=weights)
        eval_result = model.evaluate(
            data["X_train"],
            data["y_train"],
            data["X_test"],
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

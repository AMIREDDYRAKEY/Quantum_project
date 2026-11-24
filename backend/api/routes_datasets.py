"""Datasets API Endpoints"""
from flask import Blueprint, jsonify, request
from backend.core.data_manager import DataManager

datasets_bp = Blueprint("datasets", __name__, url_prefix="/api/datasets")
dm = DataManager(random_state=42)


@datasets_bp.route("/list", methods=["GET"])
def list_datasets():
    """List all available datasets and their metadata."""
    try:
        datasets = dm.list_datasets()
        return jsonify({"status": "success", "datasets": datasets})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@datasets_bp.route("/<name>/summary", methods=["GET"])
def get_dataset_summary(name):
    """Get full statistical summary, class distribution, and correlation matrix."""
    try:
        summary = dm.get_dataset_summary(name)
        return jsonify({"status": "success", "summary": summary})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 404


@datasets_bp.route("/<name>/split", methods=["POST"])
def get_dataset_split(name):
    """Perform and return data split information."""
    try:
        body = request.get_json() or {}
        test_size = float(body.get("test_size", 0.20))
        n_features = body.get("n_features")
        if n_features is not None:
            n_features = int(n_features)
        scale_method = body.get("scale_method", "standard")

        data = dm.split_and_preprocess(
            name,
            test_size=test_size,
            scale_method=scale_method,
            n_features=n_features,
        )
        return jsonify({
            "status": "success",
            "dataset_name": name,
            "n_train": data["n_train"],
            "n_test": data["n_test"],
            "total_records": data["total_records"],
            "features": data["selected_features"],
            "n_features": len(data["selected_features"]),
            "classes": data["classes"],
            "class_labels": data["class_labels"],
            "cleaning_log": data["cleaning_log"],
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

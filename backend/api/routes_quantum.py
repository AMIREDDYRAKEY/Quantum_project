"""Quantum Feature Maps and Quantum KNN API Endpoints"""
from flask import Blueprint, jsonify, request
from backend.core.data_manager import DataManager
from backend.core.quantum_knn import QuantumKNNClassifier
from backend.core.quantum_circuits import (
    get_circuit_ascii,
    get_circuit_metadata,
    FEATURE_MAP_NAMES,
)

quantum_bp = Blueprint("quantum", __name__, url_prefix="/api/quantum")
dm = DataManager(random_state=42)


@quantum_bp.route("/feature-maps", methods=["GET"])
def list_feature_maps():
    """List available quantum feature maps and metadata."""
    try:
        maps = [
            {"id": k, "name": v} for k, v in FEATURE_MAP_NAMES.items()
        ]
        return jsonify({"status": "success", "feature_maps": maps})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@quantum_bp.route("/circuit", methods=["GET"])
def get_circuit_view():
    """Get ASCII circuit diagram and gate metadata."""
    try:
        map_name = request.args.get("feature_map", "hybrid").lower()
        n_qubits = int(request.args.get("n_qubits", 4))
        depth = int(request.args.get("depth", 1))

        ascii_diagram = get_circuit_ascii(map_name, n_qubits=n_qubits, depth=depth)
        meta = get_circuit_metadata(map_name, n_qubits=n_qubits, depth=depth)

        return jsonify({
            "status": "success",
            "feature_map": map_name,
            "ascii_diagram": ascii_diagram,
            "metadata": meta,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@quantum_bp.route("/train", methods=["POST"])
def train_quantum_knn():
    """Train and evaluate baseline Quantum KNN with chosen feature map."""
    try:
        body = request.get_json() or {}
        dataset_name = body.get("dataset", "iris")
        feature_map = body.get("feature_map", "hybrid")
        k = int(body.get("k", 3))
        n_wires = int(body.get("n_wires", 4))
        depth = int(body.get("depth", 1))
        subset_size = body.get("subset_size")  # optional limit for fast interactive runs

        data = dm.split_and_preprocess(
            dataset_name,
            n_features=n_wires if dataset_name != "iris" else None,
            angle_scale=True,
        )

        X_train = data["X_train_angles"]
        y_train = data["y_train"]
        X_test = data["X_test_angles"]
        y_test = data["y_test"]

        if subset_size is not None:
            n_sub = int(subset_size)
            X_train = X_train[:n_sub]
            y_train = y_train[:n_sub]
            X_test = X_test[: min(len(X_test), max(10, n_sub // 3))]
            y_test = y_test[:len(X_test)]

        qknn = QuantumKNNClassifier(
            feature_map=feature_map,
            n_neighbors=k,
            n_wires=n_wires,
            depth=depth,
        )
        eval_result = qknn.evaluate(
            X_train, y_train, X_test, y_test, class_labels=data["class_labels"]
        )

        return jsonify({
            "status": "success",
            "dataset": dataset_name,
            "results": eval_result,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

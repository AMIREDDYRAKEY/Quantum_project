"""Model Comparison & Benchmarks API Endpoints"""
import json
from flask import Blueprint, jsonify, request
from backend.config import BENCHMARK_DIR
from backend.core.data_manager import DataManager
from backend.core.adaptive_hybrid_qknn import run_full_benchmark_comparison

benchmarks_bp = Blueprint("benchmarks", __name__, url_prefix="/api/benchmarks")
dm = DataManager(random_state=42)


@benchmarks_bp.route("/paper-results", methods=["GET"])
def get_paper_results():
    """Return published IEEE Table III results."""
    try:
        path = BENCHMARK_DIR / "paper_table3_results.json"
        if not path.exists():
            return jsonify({"status": "error", "message": "Paper results not found"}), 404
        with open(path, "r") as f:
            data = json.load(f)
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@benchmarks_bp.route("/live-results", methods=["GET"])
def get_live_results():
    """Return precomputed live benchmark results across all 4 datasets."""
    try:
        path = BENCHMARK_DIR / "live_experiment_results.json"
        if not path.exists():
            return jsonify({"status": "error", "message": "Live results not generated yet"}), 404
        with open(path, "r") as f:
            data = json.load(f)
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@benchmarks_bp.route("/run", methods=["POST"])
def run_dataset_benchmark():
    """Run full 6-model comparison on demand for selected dataset."""
    try:
        body = request.get_json() or {}
        dataset_name = body.get("dataset", "iris")
        k = int(body.get("k", 3))
        candidate_k = body.get("candidate_k", [1, 3, 5])
        n_wires = int(body.get("n_wires", 4))

        data = dm.split_and_preprocess(
            dataset_name,
            n_features=n_wires if dataset_name != "iris" else None,
            angle_scale=True,
        )

        comparison = run_full_benchmark_comparison(
            data, k=k, candidate_k=candidate_k, n_wires=n_wires
        )
        return jsonify({
            "status": "success",
            "dataset": dataset_name,
            "comparison": comparison,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

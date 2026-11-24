"""
Flask Application Factory and Server Entry Point
Enhanced Hybrid Quantum k-NN Classification Research Platform
"""
import os
import sys
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS

# Add root directory to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.api.routes_datasets import datasets_bp
from backend.api.routes_classical import classical_bp
from backend.api.routes_quantum import quantum_bp
from backend.api.routes_adaptive import adaptive_bp
from backend.api.routes_xai import xai_bp
from backend.api.routes_benchmarks import benchmarks_bp
from backend.api.routes_predict import predict_bp
from backend.api.routes_custom_dataset import dataset_custom_bp


def create_app() -> Flask:
    """Create and configure the Flask web application."""
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register API Blueprints
    app.register_blueprint(datasets_bp)
    app.register_blueprint(classical_bp)
    app.register_blueprint(quantum_bp)
    app.register_blueprint(adaptive_bp)
    app.register_blueprint(xai_bp)
    app.register_blueprint(benchmarks_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(dataset_custom_bp)

    @app.route("/", methods=["GET"])
    def index():
        return jsonify({
            "service": "Enhanced Hybrid Quantum KNN Classifier API",
            "reference_paper": "IEEE ICOCT 2025 (DOI: 10.1109/ICOCT64433.2025.11118830)",
            "simulator": "PennyLane default.qubit",
            "status": "operational",
            "endpoints": [
                "/api/health",
                "/api/datasets/list",
                "/api/classical/train",
                "/api/quantum/train",
                "/api/quantum/circuit",
                "/api/adaptive/train",
                "/api/xai/classical",
                "/api/xai/quantum",
                "/api/benchmarks/paper-results",
                "/api/benchmarks/live-results",
                "/api/predict/single",
            ],
        })

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "healthy",
            "quantum_framework": "PennyLane 0.45",
            "simulator": "default.qubit",
            "datasets_supported": ["iris", "wine", "rain", "indian_liver"],
            "models_supported": [
                "Classical_KNN",
                "Quantum_KNN_Z",
                "Quantum_KNN_ZZ",
                "Quantum_KNN_PAULI",
                "Quantum_KNN_HYBRID",
                "Adaptive_Hybrid_QKNN",
            ],
        })

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"status": "error", "message": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"status": "error", "message": "Internal server error", "detail": str(e)}), 500

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Quantum KNN Research API on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

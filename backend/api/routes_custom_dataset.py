"""
Dataset Upload, Profiling, Training, Prediction, and PDF Report API Endpoints
Compatible with both Flask and cloud deployments.
"""
from pathlib import Path
from flask import Blueprint, jsonify, request, send_file
import pandas as pd

from backend.utils.file_parser import parse_and_store_file
from backend.services.dataset_service import DatasetService
from backend.preprocessing.data_preprocessor import DataPreprocessor
from backend.services.training_service import TrainingService
from backend.reports.pdf_generator import ReportGenerator

dataset_custom_bp = Blueprint("dataset_custom", __name__, url_prefix="/api/dataset")
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


@dataset_custom_bp.route("/upload", methods=["POST"])
def upload_dataset():
    """
    Accepts CSV, XLSX, or XLS file multipart upload.
    Validates, parses into DataFrame, and returns comprehensive initial overview.
    """
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file part in request. Use 'file' field."}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"status": "error", "message": "Empty filename provided."}), 400

        file_bytes = file.read()
        dataset_id, df, filename = parse_and_store_file(file.filename, file_bytes)
        overview = DatasetService.store_dataset(dataset_id, df, filename)

        return jsonify({"status": "success", "data": overview}), 200
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Upload failed: {str(e)}"}), 500


@dataset_custom_bp.route("/load-sample/<sample_name>", methods=["POST"])
def load_sample_dataset(sample_name):
    """
    Loads pre-packaged benchmark dataset (iris, wine, rain, indian_liver).
    """
    filename_map = {
        "iris": "iris.csv",
        "wine": "wine.csv",
        "rain": "rain.csv",
        "indian_liver": "indian_liver.csv",
        "student": "student_performance.csv",
        "student_performance": "student_performance.csv",
        "agriculture": "agriculture_crop_recommendation.csv",
        "crop": "agriculture_crop_recommendation.csv",
        "social": "social_consumer_engagement.csv",
        "social_consumer": "social_consumer_engagement.csv",
    }
    sample_key = sample_name.lower().strip()
    if sample_key not in filename_map:
        return jsonify({
            "status": "error",
            "message": f"Sample '{sample_name}' not found. Available: {list(filename_map.keys())}"
        }), 404

    file_path = RAW_DATA_DIR / filename_map[sample_key]
    if not file_path.exists():
        return jsonify({"status": "error", "message": "Sample dataset file missing on server."}), 404

    df = pd.read_csv(file_path)
    dataset_id = f"sample_{sample_key}"
    overview = DatasetService.store_dataset(dataset_id, df, filename_map[sample_key])
    return jsonify({"status": "success", "data": overview}), 200


@dataset_custom_bp.route("/<dataset_id>/profile", methods=["GET"])
def get_dataset_profile(dataset_id):
    """
    Generates intelligent dataset profile:
    - Descriptive stats, distributions, correlation matrix, outlier counts
    - Automated task identification (Classification / Regression / Clustering)
    - Candidate classification target columns
    """
    try:
        profile = DatasetService.generate_profile(dataset_id)
        return jsonify({"status": "success", "data": profile}), 200
    except KeyError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": f"Profiling error: {str(e)}"}), 500


@dataset_custom_bp.route("/<dataset_id>/statistics", methods=["GET"])
def get_dataset_statistics(dataset_id):
    """Returns numerical feature statistics and correlation matrix."""
    try:
        profile = DatasetService.generate_profile(dataset_id)
        return jsonify({
            "status": "success",
            "data": {
                "dataset_id": dataset_id,
                "descriptive_statistics": profile["descriptive_statistics"],
                "correlation_matrix": profile["correlation_matrix"],
                "target_distribution": profile.get("target_distribution"),
            }
        }), 200
    except KeyError as e:
        return jsonify({"status": "error", "message": str(e)}), 404


@dataset_custom_bp.route("/<dataset_id>/configure", methods=["POST"])
def configure_dataset(dataset_id):
    """
    Configures target column, test split ratio, and qubit feature constraints.
    Applies leakage-free scaling and feature selection/PCA.
    """
    try:
        body = request.get_json() or {}
        target_column = body.get("target_column")
        if not target_column:
            return jsonify({"status": "error", "message": "Missing 'target_column' parameter."}), 400

        test_size = float(body.get("test_size", 0.20))
        max_features = int(body.get("max_features") or body.get("n_qubits") or 4)
        feature_selection_method = body.get("feature_selection_method", "f_classif")
        scale_method = body.get("scale_method") or body.get("scaling") or "standard"
        sample_size = body.get("sample_size", 200)
        random_state = int(body.get("random_state", 42))

        config_info = DataPreprocessor.configure_and_preprocess(
            dataset_id=dataset_id,
            target_column=target_column,
            test_size=test_size,
            max_features=max_features,
            feature_selection_method=feature_selection_method,
            scale_method=scale_method,
            sample_size=sample_size,
            random_state=random_state,
        )
        return jsonify({"status": "success", "data": config_info}), 200
    except KeyError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@dataset_custom_bp.route("/<dataset_id>/train", methods=["POST"])
def train_models(dataset_id):
    """
    Trains Scikit-Learn Classical KNN baseline and PennyLane Quantum KNN on default.qubit.
    Returns metrics, confusion matrices, and circuit metadata.
    """
    try:
        body = request.get_json() or {}
        k = int(body.get("k", 3))
        feature_map = body.get("feature_map", "hybrid").lower().strip()
        weighted_voting = bool(body.get("weighted_voting", True))

        results = TrainingService.train_and_evaluate(
            dataset_id=dataset_id,
            k=k,
            feature_map=feature_map,
            weighted_voting=weighted_voting,
        )
        return jsonify({"status": "success", "data": results}), 200
    except KeyError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@dataset_custom_bp.route("/<dataset_id>/results", methods=["GET"])
def get_results(dataset_id):
    """Retrieves existing benchmark training results for the dataset."""
    try:
        session = DatasetService.get_session(dataset_id)
        if not session.get("trained") or not session.get("trained_models"):
            return jsonify({"status": "error", "message": "Models have not been trained yet on this dataset."}), 400
        return jsonify({"status": "success", "data": session["trained_models"]["results"]}), 200
    except KeyError as e:
        return jsonify({"status": "error", "message": str(e)}), 404


@dataset_custom_bp.route("/<dataset_id>/predict", methods=["POST"])
def predict_single(dataset_id):
    """
    Executes live inference for custom feature input values.
    Returns both Quantum KNN and Classical KNN predictions with nearest neighbors.
    """
    try:
        body = request.get_json() or {}
        features = body.get("features", {})
        if not features:
            return jsonify({"status": "error", "message": "Missing 'features' dict in request body."}), 400

        prediction = TrainingService.predict_single(dataset_id, features)
        return jsonify({"status": "success", "data": prediction}), 200
    except KeyError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@dataset_custom_bp.route("/<dataset_id>/batch-predict", methods=["POST"])
def predict_batch(dataset_id):
    """Runs batch predictions across an array of row objects."""
    try:
        body = request.get_json() or {}
        records = body.get("records", [])
        if not records:
            return jsonify({"status": "error", "message": "Missing 'records' array in request body."}), 400

        batch_res = TrainingService.predict_batch(dataset_id, records)
        return jsonify({"status": "success", "data": batch_res}), 200
    except KeyError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@dataset_custom_bp.route("/<dataset_id>/quantum-circuit", methods=["GET"])
def get_quantum_circuit(dataset_id):
    """Returns quantum circuit diagram and gate counts."""
    try:
        session = DatasetService.get_session(dataset_id)
        if not session.get("trained"):
            return jsonify({"status": "error", "message": "Train model first to extract circuit architecture."}), 400
        trained = session["trained_models"]
        q_engine = trained["quantum_engine"]
        return jsonify({
            "status": "success",
            "data": {
                "dataset_id": dataset_id,
                "feature_map": trained["feature_map"],
                "circuit_summary": q_engine.get_circuit_metadata(),
                "ascii_diagram": q_engine.get_circuit_ascii(),
            }
        }), 200
    except KeyError as e:
        return jsonify({"status": "error", "message": str(e)}), 404


@dataset_custom_bp.route("/<dataset_id>/report", methods=["GET"])
def download_report(dataset_id):
    """Generates and serves a downloadable PDF research report."""
    try:
        pdf_path = ReportGenerator.generate_pdf_report(dataset_id)
        if not pdf_path.exists():
            return jsonify({"status": "error", "message": "Report generation failed."}), 500
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"quantum_knn_report_{dataset_id}.pdf",
            mimetype="application/pdf"
        )
    except KeyError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

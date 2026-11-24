"""
FastAPI Route Handlers
Implements all 10 endpoints for Dataset Upload, Profiling, Training, Prediction, and PDF Reporting.
"""
from typing import Dict, Any, List
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from fastapi.responses import FileResponse
import pandas as pd

from backend.schemas.dataset_schemas import (
    DatasetOverview,
    DatasetProfile,
    ConfigureDatasetRequest,
    DatasetConfigurationInfo,
    TrainModelRequest,
    TrainingResults,
    PredictSingleRequest,
    SinglePredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from backend.utils.file_parser import save_and_parse_upload
from backend.services.dataset_service import DatasetService, DATASET_STORE
from backend.preprocessing.data_preprocessor import DataPreprocessor
from backend.services.training_service import TrainingService
from backend.reports.pdf_generator import ReportGenerator

router = APIRouter(prefix="/api/dataset", tags=["Dataset Operations"])
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


@router.post("/upload", response_model=DatasetOverview)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Accepts CSV, XLSX, or XLS file.
    Validates, parses into DataFrame, and returns comprehensive initial overview.
    """
    dataset_id, df, filename = await save_and_parse_upload(file)
    overview = DatasetService.store_dataset(dataset_id, df, filename)
    return overview


@router.post("/load-sample/{sample_name}", response_model=DatasetOverview)
async def load_sample_dataset(sample_name: str):
    """
    Loads pre-packaged benchmark dataset (iris, wine, rain, indian_liver).
    """
    filename_map = {
        "iris": "iris.csv",
        "wine": "wine.csv",
        "rain": "rain.csv",
        "indian_liver": "indian_liver.csv",
    }
    sample_key = sample_name.lower().strip()
    if sample_key not in filename_map:
        raise HTTPException(
            status_code=404,
            detail=f"Sample '{sample_name}' not found. Available: {list(filename_map.keys())}"
        )

    file_path = RAW_DATA_DIR / filename_map[sample_key]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample dataset file missing on server.")

    df = pd.read_csv(file_path)
    dataset_id = f"sample_{sample_key}"
    overview = DatasetService.store_dataset(dataset_id, df, filename_map[sample_key])
    return overview


@router.get("/{dataset_id}/profile", response_model=DatasetProfile)
async def get_dataset_profile(dataset_id: str):
    """
    Generates intelligent dataset profile:
    - Descriptive stats, distributions, correlation matrix, outlier counts
    - Automated task identification (Classification / Regression / Clustering)
    - Candidate classification target columns
    """
    try:
        profile = DatasetService.generate_profile(dataset_id)
        return profile
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling error: {str(e)}")


@router.get("/{dataset_id}/statistics")
async def get_dataset_statistics(dataset_id: str):
    """Returns numerical feature statistics and correlation matrix."""
    try:
        profile = DatasetService.generate_profile(dataset_id)
        return {
            "dataset_id": dataset_id,
            "descriptive_statistics": profile["descriptive_statistics"],
            "correlation_matrix": profile["correlation_matrix"],
            "target_distribution": profile.get("target_distribution"),
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{dataset_id}/configure", response_model=DatasetConfigurationInfo)
async def configure_dataset(dataset_id: str, req: ConfigureDatasetRequest):
    """
    Configures target column, test split ratio, and qubit feature constraints.
    Applies leakage-free scaling and feature selection/PCA.
    """
    try:
        config_info = DataPreprocessor.configure_and_preprocess(
            dataset_id=dataset_id,
            target_column=req.target_column,
            test_size=req.test_size,
            max_features=req.max_features,
            feature_selection_method=req.feature_selection_method,
            scale_method=req.scale_method,
            sample_size=req.sample_size,
            random_state=req.random_state,
        )
        return config_info
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{dataset_id}/train", response_model=TrainingResults)
async def train_models(dataset_id: str, req: TrainModelRequest):
    """
    Trains Scikit-Learn Classical KNN baseline and PennyLane Quantum KNN on default.qubit.
    Returns metrics, confusion matrices, and circuit metadata.
    """
    try:
        results = TrainingService.train_and_evaluate(
            dataset_id=dataset_id,
            k=req.k,
            feature_map=req.feature_map,
            weighted_voting=req.weighted_voting,
        )
        return results
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{dataset_id}/results", response_model=TrainingResults)
async def get_results(dataset_id: str):
    """Retrieves existing benchmark training results for the dataset."""
    try:
        session = DatasetService.get_session(dataset_id)
        if not session.get("trained") or not session.get("trained_models"):
            raise HTTPException(status_code=400, detail="Models have not been trained yet on this dataset.")
        return session["trained_models"]["results"]
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{dataset_id}/predict", response_model=SinglePredictionResponse)
async def predict_single(dataset_id: str, req: PredictSingleRequest):
    """
    Executes live inference for custom feature input values.
    Returns both Quantum KNN and Classical KNN predictions with nearest neighbors.
    """
    try:
        prediction = TrainingService.predict_single(dataset_id, req.features)
        return prediction
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{dataset_id}/batch-predict", response_model=BatchPredictionResponse)
async def predict_batch(dataset_id: str, req: BatchPredictionRequest):
    """Runs batch predictions across an array of row objects."""
    try:
        batch_res = TrainingService.predict_batch(dataset_id, req.records)
        return batch_res
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{dataset_id}/quantum-circuit")
async def get_quantum_circuit(dataset_id: str):
    """Returns quantum circuit diagram and gate counts."""
    try:
        session = DatasetService.get_session(dataset_id)
        if not session.get("trained"):
            raise HTTPException(status_code=400, detail="Train model first to extract circuit architecture.")
        trained = session["trained_models"]
        q_engine = trained["quantum_engine"]
        return {
            "dataset_id": dataset_id,
            "feature_map": trained["feature_map"],
            "circuit_summary": q_engine.get_circuit_metadata(),
            "ascii_diagram": q_engine.get_circuit_ascii(),
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{dataset_id}/report")
async def download_report(dataset_id: str):
    """Generates and serves a downloadable PDF research report."""
    try:
        pdf_path = ReportGenerator.generate_pdf_report(dataset_id)
        if not pdf_path.exists():
            raise HTTPException(status_code=500, detail="Report generation failed.")
        return FileResponse(
            path=str(pdf_path),
            filename=f"quantum_knn_report_{dataset_id}.pdf",
            media_type="application/pdf"
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

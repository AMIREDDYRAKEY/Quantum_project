"""
Pydantic Schemas for Quantum KNN Research Platform
FastAPI Data Validation and Serialization
"""
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field


class DatasetOverview(BaseModel):
    dataset_id: str
    filename: str
    total_rows: int
    total_columns: int
    column_names: List[str]
    numerical_columns: List[str]
    categorical_columns: List[str]
    missing_value_count: int
    duplicate_record_count: int
    memory_usage_mb: float
    data_types: Dict[str, str]
    preview_rows: List[Dict[str, Any]]


class ColumnStatistics(BaseModel):
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[Any] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    max: Optional[float] = None
    missing_count: int = 0
    missing_percentage: float = 0.0
    unique_count: int = 0
    outlier_count: int = 0
    distribution: Optional[Dict[str, Any]] = None


class DatasetProfile(BaseModel):
    dataset_id: str
    task_identification: Dict[str, Any]
    candidate_target_columns: List[str]
    recommended_target: Optional[str] = None
    descriptive_statistics: Dict[str, ColumnStatistics]
    correlation_matrix: Dict[str, Dict[str, float]]
    target_distribution: Optional[Dict[str, int]] = None
    class_imbalance_warning: Optional[str] = None
    limitations: List[str] = []


class ConfigureDatasetRequest(BaseModel):
    target_column: str
    test_size: float = Field(default=0.20, ge=0.05, le=0.50)
    max_features: int = Field(default=4, ge=2, le=8)
    feature_selection_method: str = Field(default="f_classif")  # 'f_classif', 'mutual_info', 'pca'
    scale_method: str = Field(default="standard")  # 'standard', 'minmax'
    sample_size: Optional[int] = Field(default=200, ge=20, le=1000)
    random_state: int = Field(default=42)


class DatasetConfigurationInfo(BaseModel):
    dataset_id: str
    target_column: str
    original_feature_count: int
    selected_feature_count: int
    selected_feature_names: List[str]
    total_samples: int
    train_samples: int
    test_samples: int
    classes: List[str]
    sample_reduced: bool
    qubit_count: int
    explanation: str


class TrainModelRequest(BaseModel):
    k: int = Field(default=3, ge=1, le=15)
    feature_map: str = Field(default="hybrid")  # 'hybrid', 'pauli', 'zz', 'z', 'angle'
    weighted_voting: bool = Field(default=True)


class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    train_time_sec: float
    inference_time_sec: float
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]


class TrainingResults(BaseModel):
    dataset_id: str
    target_column: str
    classes: List[str]
    k: int
    feature_map: str
    qubits_used: int
    selected_features: List[str]
    classical_knn: ModelMetrics
    quantum_knn: ModelMetrics
    performance_comparison: Dict[str, Any]
    circuit_summary: Dict[str, Any]
    hardware_disclosure: str


class PredictSingleRequest(BaseModel):
    features: Dict[str, float]


class NeighborInfo(BaseModel):
    rank: int
    train_index: int
    class_label: str
    fidelity: Optional[float] = None
    quantum_distance: Optional[float] = None
    classical_distance: Optional[float] = None
    voting_weight: Optional[float] = None


class SinglePredictionResponse(BaseModel):
    quantum_predicted_class: str
    classical_predicted_class: str
    agreement: bool
    quantum_confidence: float
    classical_confidence: float
    nearest_neighbors: List[NeighborInfo]
    explanation: str


class BatchPredictionRequest(BaseModel):
    records: List[Dict[str, float]]


class BatchPredictionResponse(BaseModel):
    total_records: int
    predictions: List[Dict[str, Any]]

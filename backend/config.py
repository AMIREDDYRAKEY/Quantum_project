"""
Global Configuration for Enhanced Hybrid Quantum KNN Classification System
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
BENCHMARK_DIR = DATA_DIR / "benchmark_results"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

# Datasets metadata
DATASETS_CONFIG = {
    "iris": {
        "name": "Iris Dataset",
        "file": RAW_DATA_DIR / "iris.csv",
        "expected_records": 150,
        "expected_features": 4,
        "n_classes": 3,
        "target_column": "target",
        "source": "UCI Machine Learning Repository",
        "class_labels": ["setosa", "versicolor", "virginica"]
    },
    "wine": {
        "name": "Wine Dataset",
        "file": RAW_DATA_DIR / "wine.csv",
        "expected_records": 178,
        "expected_features": 13,
        "n_classes": 3,
        "target_column": "target",
        "source": "UCI Machine Learning Repository",
        "class_labels": ["Class_0", "Class_1", "Class_2"]
    },
    "rain": {
        "name": "Rain in Australia",
        "file": RAW_DATA_DIR / "rain.csv",
        "expected_records": 216,
        "expected_features": 12,
        "n_classes": 2,
        "target_column": "RainTomorrow",
        "source": "Australian Bureau of Meteorology",
        "class_labels": ["No_Rain", "Rain"]
    },
    "indian_liver": {
        "name": "Indian Liver Patient Dataset",
        "file": RAW_DATA_DIR / "indian_liver.csv",
        "expected_records": 583,
        "expected_features": 10,
        "n_classes": 2,
        "target_column": "Dataset",
        "source": "UCI Machine Learning Repository",
        "class_labels": ["Non_Patient", "Patient"]
    }
}

# Preprocessing & Model Defaults
DEFAULT_RANDOM_SEED = 42
DEFAULT_TEST_SIZE = 0.20  # 80:20 train-test ratio
DEFAULT_K = 3
DEFAULT_CIRCUIT_WIRES = 4  # Matches 4-qubit circuit schematics from paper
SIMULATOR_DEVICE = "default.qubit"

"""
Unit tests for Dataset Management and Preprocessing (Phase 3)
Verifies:
- All 4 canonical datasets (Iris, Wine, Rain, Indian Liver)
- Dimensions, features, records, and class counts matching Table II of IEEE paper
- Missing value imputation and duplicate reporting
- Leakage-free StandardScaler fitted strictly on train data
- Stratified 80:20 train-test splitting
- Angle scaling for quantum gates (bounded in [-pi, pi])
- Feature selection for 4-qubit circuit matching
"""
import pytest
import numpy as np
import pandas as pd
from backend.core.data_manager import DataManager
from backend.config import DATASETS_CONFIG


@pytest.fixture
def dm():
    return DataManager(random_state=42)


def test_datasets_exist_and_metadata(dm):
    """Test that all 4 datasets are registered and raw files exist."""
    datasets = dm.list_datasets()
    assert len(datasets) == 4
    dataset_ids = [d["id"] for d in datasets]
    assert set(dataset_ids) == {"iris", "wine", "rain", "indian_liver"}
    for d in datasets:
        assert d["available"] is True, f"Dataset {d['id']} raw file missing"


@pytest.mark.parametrize("dataset_name,expected_records,expected_features,expected_classes", [
    ("iris", 150, 4, 3),
    ("wine", 178, 13, 3),
    ("rain", 216, 12, 2),
    ("indian_liver", 583, 10, 2),
])
def test_dataset_dimensions_match_ieee_paper(
    dm, dataset_name, expected_records, expected_features, expected_classes
):
    """Verify exact records, features, and classes match Table II of IEEE paper."""
    conf = DATASETS_CONFIG[dataset_name]
    df = dm.load_dataset(dataset_name)
    assert len(df) == expected_records, (
        f"{dataset_name} records mismatch: got {len(df)}, expected {expected_records}"
    )
    # Exclude target column for feature count
    target_col = conf["target_column"]
    assert target_col in df.columns
    features = [c for c in df.columns if c != target_col]
    assert len(features) == expected_features, (
        f"{dataset_name} feature count mismatch: got {len(features)}, expected {expected_features}"
    )
    unique_targets = df[target_col].dropna().nunique()
    assert unique_targets == expected_classes, (
        f"{dataset_name} classes mismatch: got {unique_targets}, expected {expected_classes}"
    )


def test_missing_value_imputation(dm):
    """Verify that Indian Liver missing values in Albumin_and_Globulin_Ratio are properly imputed."""
    raw_df = dm.load_dataset("indian_liver")
    # Check that raw data has missing values
    missing_count = raw_df["Albumin_and_Globulin_Ratio"].isna().sum()
    assert missing_count == 4, f"Expected 4 missing values in raw Indian Liver, got {missing_count}"

    clean_df, log = dm.clean_and_encode(raw_df, "Dataset")
    assert clean_df["Albumin_and_Globulin_Ratio"].isna().sum() == 0
    assert log["missing_values_detected"] == 4


def test_stratified_80_20_split_ratios(dm):
    """Verify 80:20 train-test stratified split for all 4 datasets."""
    expected_splits = {
        "iris": (120, 30),
        "wine": (142, 36),
        "rain": (172, 44),
        "indian_liver": (466, 117),
    }

    for name, (exp_train, exp_test) in expected_splits.items():
        data = dm.split_and_preprocess(name, test_size=0.20, random_state=42)
        assert data["n_train"] == exp_train, f"{name} train size {data['n_train']} != {exp_train}"
        assert data["n_test"] == exp_test, f"{name} test size {data['n_test']} != {exp_test}"
        assert data["total_records"] == exp_train + exp_test

        # Verify stratification preserves class balance
        train_class_ratios = pd.Series(data["y_train"]).value_counts(normalize=True).sort_index()
        test_class_ratios = pd.Series(data["y_test"]).value_counts(normalize=True).sort_index()
        for c in train_class_ratios.index:
            assert abs(train_class_ratios[c] - test_class_ratios[c]) < 0.05, (
                f"Stratification imbalance in {name} class {c}"
            )


def test_zero_data_leakage_scaling(dm):
    """
    Verify that StandardScaler is fitted strictly on X_train.
    Train mean should be ~0, train std should be ~1.
    Test mean and std should reflect transform with train parameters.
    """
    data = dm.split_and_preprocess("wine", scale_method="standard")
    X_train = data["X_train"]
    X_test = data["X_test"]

    # Train set should have mean ~0 and std ~1 along features
    assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(X_train.std(axis=0), 1.0, atol=1e-7)

    # Test set transformed using train scaler should not be identically 0 and 1
    test_means = X_test.mean(axis=0)
    assert not np.allclose(test_means, 0.0, atol=1e-7)


def test_quantum_angle_scaling(dm):
    """Verify that angle scaling restricts values to [-pi, pi]."""
    for name in ["iris", "wine", "rain", "indian_liver"]:
        data = dm.split_and_preprocess(name, angle_scale=True)
        X_train_angles = data["X_train_angles"]
        X_test_angles = data["X_test_angles"]

        assert np.all(X_train_angles >= -np.pi) and np.all(X_train_angles <= np.pi)
        assert np.all(X_test_angles >= -np.pi) and np.all(X_test_angles <= np.pi)


def test_feature_selection_to_4_qubits(dm):
    """
    Verify feature selection reduces Wine (13), Rain (12), Liver (10)
    to exactly 4 features for the 4-qubit circuit schematics.
    """
    for name in ["wine", "rain", "indian_liver"]:
        data = dm.split_and_preprocess(name, n_features=4, feature_selection_method="f_classif")
        assert data["X_train"].shape[1] == 4
        assert data["X_test"].shape[1] == 4
        assert len(data["selected_features"]) == 4


def test_dataset_summary_statistics(dm):
    """Verify statistical summary generation for API."""
    summary = dm.get_dataset_summary("iris")
    assert summary["id"] == "iris"
    assert summary["total_records"] == 150
    assert summary["n_features"] == 4
    assert summary["n_classes"] == 3
    assert len(summary["preview_rows"]) == 10
    assert "sepal_length" in summary["feature_statistics"]
    assert "mean" in summary["feature_statistics"]["sepal_length"]

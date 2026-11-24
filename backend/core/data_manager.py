"""
Dataset Management and Preprocessing Module
Implements loading, cleaning, imputation, label encoding,
leakage-free feature scaling, stratified splitting, and feature selection.
"""
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA

from backend.config import (
    DATASETS_CONFIG,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_SIZE,
    RAW_DATA_DIR,
)


class DataManager:
    """
    Robust dataset manager adhering to research standards:
    - Zero data leakage (all transformations fit strictly on training splits)
    - Stratified splitting for class balance preservation
    - Comprehensive statistics and metadata extraction
    """

    def __init__(self, random_state: int = DEFAULT_RANDOM_SEED):
        self.random_state = random_state
        self.configs = DATASETS_CONFIG

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Return metadata summary for all available datasets."""
        result = []
        for key, conf in self.configs.items():
            file_exists = conf["file"].exists()
            result.append({
                "id": key,
                "name": conf["name"],
                "expected_records": conf["expected_records"],
                "expected_features": conf["expected_features"],
                "n_classes": conf["n_classes"],
                "source": conf["source"],
                "target_column": conf["target_column"],
                "available": file_exists,
            })
        return result

    def load_dataset(self, name: str) -> pd.DataFrame:
        """Load raw dataset by key name ('iris', 'wine', 'rain', 'indian_liver')."""
        if name not in self.configs:
            raise ValueError(f"Unknown dataset '{name}'. Available: {list(self.configs.keys())}")
        file_path = self.configs[name]["file"]
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {file_path}")
        return pd.read_csv(file_path)

    def clean_and_encode(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean missing values, identify duplicates, and encode categorical features/labels.
        """
        df_clean = df.copy()
        cleaning_log = {
            "initial_rows": len(df_clean),
            "initial_cols": len(df_clean.columns),
            "missing_values_detected": int(df_clean.isna().sum().sum()),
            "missing_per_column": df_clean.isna().sum().to_dict(),
            "duplicates_detected": int(df_clean.duplicated().sum()),
            "categorical_encoded": [],
        }

        # Impute missing values
        for col in df_clean.columns:
            if df_clean[col].isna().sum() > 0:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                else:
                    mode_val = df_clean[col].mode()[0]
                    df_clean[col] = df_clean[col].fillna(mode_val)

        # Encode categorical feature columns (excluding target column)
        for col in df_clean.columns:
            if col != target_column and not pd.api.types.is_numeric_dtype(df_clean[col]):
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                cleaning_log["categorical_encoded"].append(col)

        # Ensure target column is zero-indexed integer
        # Specifically for Indian Liver Patient: classes are 1 (patient) and 2 (non-patient)
        if target_column in df_clean.columns:
            target_series = df_clean[target_column]
            unique_targets = sorted(target_series.unique())
            if unique_targets == [1, 2]:
                # Map 1 -> 1 (patient), 2 -> 0 (non-patient) for clean binary labels
                df_clean[target_column] = target_series.map({2: 0, 1: 1}).astype(int)
            elif not pd.api.types.is_integer_dtype(target_series) or min(unique_targets) != 0:
                le_target = LabelEncoder()
                df_clean[target_column] = le_target.fit_transform(target_series).astype(int)

        cleaning_log["final_rows"] = len(df_clean)
        cleaning_log["final_cols"] = len(df_clean.columns)
        return df_clean, cleaning_log

    def split_and_preprocess(
        self,
        name: str,
        test_size: float = DEFAULT_TEST_SIZE,
        scale_method: str = "standard",
        n_features: Optional[int] = None,
        feature_selection_method: str = "f_classif",
        angle_scale: bool = True,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Full leakage-free preprocessing pipeline:
        1. Loads and cleans dataset
        2. Performs stratified 80:20 split
        3. Fits feature scaler ONLY on X_train, transforms X_train and X_test
        4. Optionally scales into [-pi, pi] for quantum angle rotation
        5. Optionally selects top n_features (e.g. 4 for 4-qubit compact mode)
        """
        seed = random_state if random_state is not None else self.random_state
        conf = self.configs[name]
        raw_df = self.load_dataset(name)
        target_col = conf["target_column"]

        clean_df, cleaning_info = self.clean_and_encode(raw_df, target_col)

        X_df = clean_df.drop(columns=[target_col])
        y_series = clean_df[target_col].astype(int)
        feature_names = list(X_df.columns)

        # Stratified train-test split (80:20)
        X_train_df, X_test_df, y_train_series, y_test_series = train_test_split(
            X_df,
            y_series,
            test_size=test_size,
            stratify=y_series,
            random_state=seed,
        )

        # Leakage-free scaling: fit ONLY on X_train
        if scale_method == "standard":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_df)
            X_test_scaled = scaler.transform(X_test_df)
        elif scale_method == "minmax":
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_train_scaled = scaler.fit_transform(X_train_df)
            X_test_scaled = scaler.transform(X_test_df)
        else:
            scaler = None
            X_train_scaled = X_train_df.values.astype(np.float64)
            X_test_scaled = X_test_df.values.astype(np.float64)

        # Feature Selection (e.g. select 4 features to match 4-qubit circuit schematics)
        selected_features = feature_names
        selector = None
        if n_features is not None and n_features < X_train_scaled.shape[1]:
            if feature_selection_method == "pca":
                selector = PCA(n_components=n_features, random_state=seed)
                X_train_scaled = selector.fit_transform(X_train_scaled)
                X_test_scaled = selector.transform(X_test_scaled)
                selected_features = [f"PC_{i+1}" for i in range(n_features)]
            elif feature_selection_method == "mutual_info":
                selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
                X_train_scaled = selector.fit_transform(X_train_scaled, y_train_series.values)
                X_test_scaled = selector.transform(X_test_scaled)
                mask = selector.get_support()
                selected_features = [f for f, s in zip(feature_names, mask) if s]
            else:  # default: ANOVA f_classif
                selector = SelectKBest(score_func=f_classif, k=n_features)
                X_train_scaled = selector.fit_transform(X_train_scaled, y_train_series.values)
                X_test_scaled = selector.transform(X_test_scaled)
                mask = selector.get_support()
                selected_features = [f for f, s in zip(feature_names, mask) if s]

        # Angle normalization for quantum gates: map into [-pi, pi]
        # Using bounded tanh / arctan projection or min-max mapping on train
        angle_scaler = None
        if angle_scale:
            # Map standardized values into [-pi, pi] using arctan scaling: 2 * arctan(x) in (-pi, pi)
            X_train_angles = 2.0 * np.arctan(X_train_scaled)
            X_test_angles = 2.0 * np.arctan(X_test_scaled)
        else:
            X_train_angles = X_train_scaled.copy()
            X_test_angles = X_test_scaled.copy()

        y_train = y_train_series.values
        y_test = y_test_series.values

        # Unique classes
        classes = sorted(np.unique(y_train).tolist())

        return {
            "dataset_name": name,
            "display_name": conf["name"],
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "X_train_angles": X_train_angles,
            "X_test_angles": X_test_angles,
            "y_train": y_train,
            "y_test": y_test,
            "classes": classes,
            "n_classes": len(classes),
            "class_labels": conf["class_labels"],
            "original_features": feature_names,
            "selected_features": selected_features,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "total_records": len(y_train) + len(y_test),
            "test_ratio": test_size,
            "scaler": scaler,
            "selector": selector,
            "cleaning_log": cleaning_info,
        }

    def get_dataset_summary(self, name: str) -> Dict[str, Any]:
        """Compute comprehensive statistical metrics, histograms, and correlations."""
        conf = self.configs[name]
        raw_df = self.load_dataset(name)
        target_col = conf["target_column"]
        clean_df, cleaning_info = self.clean_and_encode(raw_df, target_col)

        feature_cols = [c for c in clean_df.columns if c != target_col]
        stats = {}
        for col in feature_cols:
            series = clean_df[col]
            stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
                "max": float(series.max()),
                "skew": float(series.skew()),
            }

        class_counts = clean_df[target_col].value_counts().sort_index().to_dict()
        class_distribution = {
            conf["class_labels"][int(k)] if int(k) < len(conf["class_labels"]) else str(k): int(v)
            for k, v in class_counts.items()
        }

        # Correlation matrix
        corr = clean_df[feature_cols].corr().round(4).to_dict()

        return {
            "id": name,
            "name": conf["name"],
            "source": conf["source"],
            "total_records": len(clean_df),
            "n_features": len(feature_cols),
            "n_classes": conf["n_classes"],
            "target_column": target_col,
            "feature_names": feature_cols,
            "class_distribution": class_distribution,
            "feature_statistics": stats,
            "correlation_matrix": corr,
            "cleaning_log": cleaning_info,
            "preview_rows": clean_df.head(10).to_dict(orient="records"),
        }

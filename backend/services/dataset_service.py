"""
Dataset Management, Profiling, and Task Identification Service
Analyzes uploaded tabular datasets and detects machine learning suitability.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from backend.utils.file_parser import UPLOAD_DIR

# Global in-memory dataset session store
DATASET_STORE: Dict[str, Dict[str, Any]] = {}


class DatasetService:
    @staticmethod
    def store_dataset(dataset_id: str, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Cache DataFrame and metadata in store."""
        DATASET_STORE[dataset_id] = {
            "id": dataset_id,
            "filename": filename,
            "df": df,
            "configured": False,
            "trained": False,
            "preprocessed_data": None,
            "trained_models": None,
            "report_path": None,
        }
        return DatasetService.get_overview(dataset_id)

    @staticmethod
    def get_dataset(dataset_id: str) -> pd.DataFrame:
        if dataset_id not in DATASET_STORE:
            raise KeyError(f"Dataset '{dataset_id}' not found in active session.")
        return DATASET_STORE[dataset_id]["df"]

    @staticmethod
    def get_session(dataset_id: str) -> Dict[str, Any]:
        if dataset_id not in DATASET_STORE:
            raise KeyError(f"Dataset '{dataset_id}' not found in active session.")
        return DATASET_STORE[dataset_id]

    @staticmethod
    def get_overview(dataset_id: str) -> Dict[str, Any]:
        """Extract clean dataset statistics and schema preview."""
        df = DatasetService.get_dataset(dataset_id)
        session = DATASET_STORE[dataset_id]

        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(exclude=[np.number]).columns)

        missing_count = int(df.isna().sum().sum())
        duplicate_count = int(df.duplicated().sum())
        memory_mb = round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 3)

        data_types = {col: str(df[col].dtype) for col in df.columns}
        preview = df.head(10).replace({np.nan: None}).to_dict(orient="records")

        return {
            "dataset_id": dataset_id,
            "filename": session["filename"],
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": list(df.columns),
            "numerical_columns": num_cols,
            "categorical_columns": cat_cols,
            "missing_value_count": missing_count,
            "duplicate_record_count": duplicate_count,
            "memory_usage_mb": memory_mb,
            "data_types": data_types,
            "preview_rows": preview,
        }

    @staticmethod
    def generate_profile(dataset_id: str) -> Dict[str, Any]:
        """
        Comprehensive automated profiling:
        - Descriptive statistics for all features
        - Correlation matrix
        - Outlier detection (IQR method)
        - Automated Task Identification (Classification / Regression / Clustering)
        - Target candidate detection
        - Clear scientific disclosures and limitations
        """
        df = DatasetService.get_dataset(dataset_id)
        n_rows, n_cols = df.shape

        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(exclude=[np.number]).columns)

        stats_dict = {}
        candidate_targets = []
        limitations = []

        # 1. Descriptive statistics & outlier detection
        for col in df.columns:
            series = df[col]
            missing = int(series.isna().sum())
            missing_pct = round((missing / n_rows) * 100, 2)
            n_unique = int(series.nunique(dropna=True))

            # Candidate target detection heuristics
            if 2 <= n_unique <= 20 and missing_pct < 20:
                candidate_targets.append(col)
            elif col in cat_cols and n_unique <= 30:
                candidate_targets.append(col)

            if col in num_cols:
                valid_num = series.dropna()
                if len(valid_num) > 0:
                    q25 = float(valid_num.quantile(0.25))
                    q75 = float(valid_num.quantile(0.75))
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    outliers = int(((valid_num < lower_bound) | (valid_num > upper_bound)).sum())

                    # 10-bin histogram for UI distribution chart
                    counts, bin_edges = np.histogram(valid_num, bins=min(10, max(3, n_unique)))
                    hist_data = [
                        {"bin": f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}", "count": int(counts[i])}
                        for i in range(len(counts))
                    ]

                    stats_dict[col] = {
                        "mean": round(float(valid_num.mean()), 4),
                        "median": round(float(valid_num.median()), 4),
                        "mode": float(valid_num.mode()[0]) if len(valid_num.mode()) > 0 else None,
                        "std": round(float(valid_num.std()), 4) if len(valid_num) > 1 else 0.0,
                        "min": round(float(valid_num.min()), 4),
                        "q25": round(q25, 4),
                        "q75": round(q75, 4),
                        "max": round(float(valid_num.max()), 4),
                        "missing_count": missing,
                        "missing_percentage": missing_pct,
                        "unique_count": n_unique,
                        "outlier_count": outliers,
                        "distribution": hist_data,
                    }
                else:
                    stats_dict[col] = {
                        "missing_count": missing,
                        "missing_percentage": missing_pct,
                        "unique_count": 0,
                        "outlier_count": 0,
                    }
            else:
                # Categorical column summary
                top_cats = series.value_counts(dropna=True).head(5).to_dict()
                stats_dict[col] = {
                    "mean": None,
                    "median": None,
                    "mode": str(series.mode()[0]) if len(series.mode()) > 0 else None,
                    "std": None,
                    "min": None,
                    "q25": None,
                    "q75": None,
                    "max": None,
                    "missing_count": missing,
                    "missing_percentage": missing_pct,
                    "unique_count": n_unique,
                    "outlier_count": 0,
                    "distribution": [{"bin": str(k), "count": int(v)} for k, v in top_cats.items()],
                }

        # 2. Correlation matrix (for numerical columns)
        if len(num_cols) > 1:
            corr_df = df[num_cols].corr().round(4).fillna(0.0)
            corr_matrix = corr_df.to_dict()
        else:
            corr_matrix = {}

        # 3. Task Identification Logic
        # Try to guess default recommended target (e.g. columns named 'target', 'class', 'label', 'species')
        recommended_target = None
        for cand in candidate_targets:
            if cand.lower() in ["target", "class", "label", "species", "category", "outcome", "diagnosis"]:
                recommended_target = cand
                break
        if not recommended_target and len(candidate_targets) > 0:
            recommended_target = candidate_targets[-1]  # Typically last column in CSV

        task_info = {}
        target_dist = None
        class_imbalance = None

        if recommended_target:
            target_series = df[recommended_target].dropna()
            n_classes = target_series.nunique()
            counts = target_series.value_counts().to_dict()
            target_dist = {str(k): int(v) for k, v in counts.items()}

            if n_classes >= 2:
                task_info = {
                    "task_type": "Classification",
                    "subtype": "Binary Classification" if n_classes == 2 else "Multiclass Classification",
                    "num_classes": n_classes,
                    "suitable_for_qknn": True,
                    "reason": f"Target column '{recommended_target}' has {n_classes} discrete categories suitable for Supervised Quantum KNN classification.",
                }

                # Check class imbalance
                min_ratio = min(counts.values()) / sum(counts.values())
                if min_ratio < 0.20:
                    class_imbalance = f"Warning: Severe class imbalance detected. Minority class represents only {min_ratio*100:.1f}% of samples."
            else:
                task_info = {
                    "task_type": "Degenerate Target",
                    "suitable_for_qknn": False,
                    "reason": f"Column '{recommended_target}' has only {n_classes} distinct class. Supervised classification requires at least 2 distinct classes.",
                }
                limitations.append("Classification requires at least 2 target classes.")
        elif len(num_cols) == len(df.columns) and len(num_cols) > 1:
            task_info = {
                "task_type": "Regression / Clustering / Statistical Analysis",
                "suitable_for_qknn": False,
                "reason": "All columns appear to be continuous numerical features. The Quantum KNN classifier is currently implemented for Supervised Classification only.",
            }
            limitations.append("Continuous regression targets are not directly supported by this QKNN classification model.")
            limitations.append("Unsupervised clustering analysis can be explored, but QKNN classification requires a labeled target.")
        else:
            task_info = {
                "task_type": "General Statistical Analysis",
                "suitable_for_qknn": False,
                "reason": "No discrete classification target column was automatically identifiable. Please select or provide a labeled target column.",
            }
            limitations.append("No automatic classification target detected. Select a target column to proceed.")

        # Computational limit warning for quantum simulation
        if n_cols > 8:
            limitations.append(f"Uploaded dataset contains {n_cols} columns. Quantum simulation scales exponentially ($2^N$ amplitudes), so feature selection/PCA will select the top 4 to 8 features for qubit execution.")
        if n_rows > 300:
            limitations.append(f"Uploaded dataset has {n_rows} rows. To maintain responsive quantum simulator runtimes, stratified sampling will be applied.")

        return {
            "dataset_id": dataset_id,
            "task_identification": task_info,
            "candidate_target_columns": candidate_targets,
            "recommended_target": recommended_target,
            "descriptive_statistics": stats_dict,
            "correlation_matrix": corr_matrix,
            "target_distribution": target_dist,
            "class_imbalance_warning": class_imbalance,
            "limitations": limitations,
        }

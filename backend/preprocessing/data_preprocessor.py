"""
Comprehensive Tabular Preprocessing Pipeline
Leakage-free scaling, categorical encoding, feature selection, and quantum angle mapping.
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from backend.services.dataset_service import DatasetService


class DataPreprocessor:
    @staticmethod
    def configure_and_preprocess(
        dataset_id: str,
        target_column: str,
        test_size: float = 0.20,
        max_features: int = 4,
        feature_selection_method: str = "f_classif",
        scale_method: str = "standard",
        sample_size: Optional[int] = 200,
        random_state: int = 42,
        selected_features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete end-to-end preprocessing:
        1. Separate features and target (exclude target column from features)
        2. Filter by user-selected features if specified
        3. Impute missing values and encode categorical features
        4. Optional stratified subsampling for quantum runtime feasibility
        5. Stratified train-test split (zero data leakage)
        6. Fit scaler on X_train only, transform X_test
        7. Feature selection / PCA down to max_features (qubit count)
        8. Map to quantum rotation angles [-pi, pi]
        """
        df = DatasetService.get_dataset(dataset_id).copy()
        original_shape = [int(len(df)), int(len(df.columns))]
        warnings_list: List[str] = []

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' does not exist in dataset.")

        # Drop rows where target is missing
        n_missing_target = df[target_column].isna().sum()
        if n_missing_target > 0:
            warnings_list.append(f"Dropped {n_missing_target} records with missing target values.")
            df = df.dropna(subset=[target_column])

        # Separate X and y
        y_raw = df[target_column]
        X_raw = df.drop(columns=[target_column])

        # Filter to user-selected input features if specified
        if selected_features and isinstance(selected_features, list) and len(selected_features) > 0:
            valid_feats = [col for col in selected_features if col in X_raw.columns and col != target_column]
            if len(valid_feats) > 0:
                X_raw = X_raw[valid_feats]
            else:
                warnings_list.append("Specified features not found; defaulted to all available predictive features.")

        # Encode target classes
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_raw)
        class_names = [str(c) for c in label_encoder.classes_]
        n_classes = len(class_names)

        if n_classes < 2:
            raise ValueError("Target column must contain at least 2 distinct classes.")

        # Drop identifier columns (high cardinality string columns)
        cols_to_drop = []
        for col in X_raw.columns:
            if X_raw[col].dtype == object and X_raw[col].nunique() == len(X_raw):
                cols_to_drop.append(col)
        if cols_to_drop:
            warnings_list.append(f"Excluded high-cardinality ID columns: {', '.join(cols_to_drop)}.")
            X_raw = X_raw.drop(columns=cols_to_drop)

        # Impute missing values & encode categorical features
        X_clean = pd.DataFrame(index=X_raw.index)
        for col in X_raw.columns:
            series = X_raw[col]
            if pd.api.types.is_numeric_dtype(series):
                med = series.median()
                X_clean[col] = series.fillna(med if not pd.isna(med) else 0.0)
            else:
                # Mode impute and integer code
                mode_val = series.mode()[0] if len(series.mode()) > 0 else "missing"
                filled = series.fillna(mode_val).astype(str)
                X_clean[col] = LabelEncoder().fit_transform(filled)

        # Drop constant columns (zero variance)
        constant_cols = [c for c in X_clean.columns if X_clean[c].nunique() <= 1]
        if constant_cols:
            X_clean = X_clean.drop(columns=constant_cols)

        if X_clean.empty:
            raise ValueError("No valid predictive features remaining after data cleaning.")

        original_feature_names = list(X_clean.columns)
        total_samples = len(X_clean)

        # Stratified subsampling for computationally intensive quantum simulation
        sample_reduced = False
        if sample_size and total_samples > sample_size:
            sample_reduced = True
            X_clean, _, y_encoded, _ = train_test_split(
                X_clean,
                y_encoded,
                train_size=sample_size,
                stratify=y_encoded,
                random_state=random_state,
            )

        # Stratified train/test split (Zero Data Leakage)
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_clean,
            y_encoded,
            test_size=test_size,
            stratify=y_encoded,
            random_state=random_state,
        )

        # Leakage-Free Scaling (fit ONLY on X_train)
        if scale_method == "minmax":
            scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train_df)
        X_test_scaled = scaler.transform(X_test_df)

        # Feature Selection / PCA to match Qubit constraints (e.g. 4 to 8 qubits)
        n_orig_features = X_train_scaled.shape[1]
        target_features = min(max_features, n_orig_features)

        selector_model = None
        selected_feature_names = original_feature_names.copy()

        if n_orig_features > target_features:
            if feature_selection_method == "pca":
                selector_model = PCA(n_components=target_features, random_state=random_state)
                X_train_reduced = selector_model.fit_transform(X_train_scaled)
                X_test_reduced = selector_model.transform(X_test_scaled)
                selected_feature_names = [f"PC_{i+1}" for i in range(target_features)]
            elif feature_selection_method == "mutual_info":
                selector_model = SelectKBest(mutual_info_classif, k=target_features)
                X_train_reduced = selector_model.fit_transform(X_train_scaled, y_train)
                X_test_reduced = selector_model.transform(X_test_scaled)
                mask = selector_model.get_support()
                selected_feature_names = [name for name, m in zip(original_feature_names, mask) if m]
            else:  # Default: f_classif (ANOVA)
                selector_model = SelectKBest(f_classif, k=target_features)
                X_train_reduced = selector_model.fit_transform(X_train_scaled, y_train)
                X_test_reduced = selector_model.transform(X_test_scaled)
                mask = selector_model.get_support()
                selected_feature_names = [name for name, m in zip(original_feature_names, mask) if m]
        else:
            X_train_reduced = X_train_scaled
            X_test_reduced = X_test_scaled

        # Quantum Angle Mapping: map values into [-pi, pi] for Pauli rotations
        X_train_angles = 2.0 * np.arctan(X_train_reduced)
        X_test_angles = 2.0 * np.arctan(X_test_reduced)

        explanation = (
            f"Preprocessed {total_samples} samples. "
            f"Fitted {scale_method.capitalize()}Scaler exclusively on {len(y_train)} training instances. "
        )
        if n_orig_features > target_features:
            explanation += f"Selected top {target_features} features from {n_orig_features} using {feature_selection_method.upper()} to map to {target_features} qubits."
        else:
            explanation += f"Directly mapped {target_features} features to {target_features} qubits."

        processed_state = {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "classes": class_names,
            "n_classes": n_classes,
            "label_encoder": label_encoder,
            "scaler": scaler,
            "selector": selector_model,
            "scale_method": scale_method,
            "feature_selection_method": feature_selection_method,
            "original_features": original_feature_names,
            "selected_features": selected_feature_names,
            "n_qubits": target_features,
            "total_samples": total_samples,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "original_shape": original_shape,
            "processed_shape": [total_samples, target_features],
            "warnings": warnings_list,
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "X_train_reduced": X_train_reduced,
            "X_test_reduced": X_test_reduced,
            "X_train_angles": X_train_angles,
            "X_test_angles": X_test_angles,
            "y_train": y_train,
            "y_test": y_test,
            "sample_reduced": sample_reduced,
            "explanation": explanation,
        }

        # Cache in global dataset store
        session = DatasetService.get_session(dataset_id)
        session["configured"] = True
        session["preprocessed_data"] = processed_state

        return {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "original_shape": original_shape,
            "processed_shape": [total_samples, target_features],
            "original_feature_count": n_orig_features,
            "selected_feature_count": target_features,
            "selected_feature_names": selected_feature_names,
            "total_samples": total_samples,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "classes": class_names,
            "number_of_classes": n_classes,
            "scale_method": scale_method,
            "sample_reduced": sample_reduced,
            "qubit_count": target_features,
            "warnings": warnings_list,
            "explanation": explanation,
        }

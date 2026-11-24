"""
Verification script for FastAPI pipeline without HTTP network stack.
Directly tests service layer functions.
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from backend.services.dataset_service import DatasetService
from backend.preprocessing.data_preprocessor import DataPreprocessor
from backend.services.training_service import TrainingService
from backend.reports.pdf_generator import ReportGenerator

print("--- Testing FastAPI Backend Services ---")

# 1. Load Iris
iris_path = ROOT_DIR / "backend" / "data" / "raw" / "iris.csv"
df = pd.read_csv(iris_path)
dataset_id = "test_iris"
overview = DatasetService.store_dataset(dataset_id, df, "iris.csv")
print(f"[1/5] Ingested: {overview['filename']}, Rows: {overview['total_rows']}, Cols: {overview['total_columns']}")

# 2. Profile
profile = DatasetService.generate_profile(dataset_id)
task = profile["task_identification"]
print(f"[2/5] Task Identified: {task['task_type']} ({task.get('subtype')}), Recommended Target: {profile['recommended_target']}")

# 3. Configure & Preprocess
config = DataPreprocessor.configure_and_preprocess(
    dataset_id=dataset_id,
    target_column=profile["recommended_target"],
    test_size=0.20,
    max_features=4,
    feature_selection_method="f_classif",
    scale_method="standard",
)
print(f"[3/5] Preprocessed: Qubits={config['qubit_count']}, Train={config['train_samples']}, Test={config['test_samples']}, Features={config['selected_feature_names']}")

# 4. Train Classical vs Quantum KNN
t_res = TrainingService.train_and_evaluate(
    dataset_id=dataset_id,
    k=3,
    feature_map="hybrid",
    weighted_voting=True
)
c_acc = t_res["classical_knn"]["accuracy"] * 100
q_acc = t_res["quantum_knn"]["accuracy"] * 100
print(f"[4/5] Trained: Classical KNN Acc={c_acc:.1f}%, Quantum KNN Acc={q_acc:.1f}%, Gates={t_res['circuit_summary']['total_gates']}, CNOTs={t_res['circuit_summary']['cnot_gates']}")

# 5. Predict
pred = TrainingService.predict_single(dataset_id, {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
print(f"[5/5] Prediction: Q-Pred={pred['quantum_predicted_class']}, C-Pred={pred['classical_predicted_class']}, Agreement={pred['agreement']}, Top Neighbor Fidelity={pred['nearest_neighbors'][0]['fidelity']}")

# 6. PDF Report
pdf_path = ReportGenerator.generate_pdf_report(dataset_id)
print(f"[6/6] PDF Report Generated: {pdf_path.name} ({pdf_path.stat().st_size} bytes)")

print("\n>>> ALL BACKEND SERVICES VERIFIED SUCCESSFULLY! <<<")

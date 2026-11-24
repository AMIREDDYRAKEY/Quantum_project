import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app import app

client = app.test_client()

print("--- Testing Dataset Custom Endpoints on App ---")

# 1. Load sample dataset
r1 = client.post("/api/dataset/load-sample/iris")
assert r1.status_code == 200, f"Load sample failed: {r1.get_json()}"
d1 = r1.get_json()
dataset_id = d1["data"]["dataset_id"]
print(f"[1/6] Loaded: {d1['data']['filename']}, ID: {dataset_id}, Rows: {d1['data']['total_rows']}")

# 2. Profile
r2 = client.get(f"/api/dataset/{dataset_id}/profile")
assert r2.status_code == 200, f"Profile failed: {r2.get_json()}"
d2 = r2.get_json()["data"]
print(f"[2/6] Profile Task: {d2['task_identification']['task_type']}, Target: {d2['recommended_target']}")

# 3. Configure
r3 = client.post(f"/api/dataset/{dataset_id}/configure", json={
    "target_column": d2["recommended_target"],
    "test_size": 0.20,
    "max_features": 4,
    "feature_selection_method": "f_classif",
    "scale_method": "standard",
})
assert r3.status_code == 200, f"Configure failed: {r3.get_json()}"
d3 = r3.get_json()["data"]
print(f"[3/6] Configured: Qubits={d3['qubit_count']}, Train={d3['train_samples']}, Test={d3['test_samples']}")

# 4. Train
r4 = client.post(f"/api/dataset/{dataset_id}/train", json={
    "k": 3,
    "feature_map": "hybrid",
    "weighted_voting": True,
})
assert r4.status_code == 200, f"Train failed: {r4.get_json()}"
d4 = r4.get_json()["data"]
c_acc = d4["classical_knn"]["accuracy"] * 100
q_acc = d4["quantum_knn"]["accuracy"] * 100
print(f"[4/6] Trained: Classical={c_acc:.1f}%, Quantum={q_acc:.1f}%")

# 5. Predict
r5 = client.post(f"/api/dataset/{dataset_id}/predict", json={
    "features": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
})
assert r5.status_code == 200, f"Predict failed: {r5.get_json()}"
d5 = r5.get_json()["data"]
print(f"[5/6] Prediction: Q-Class={d5['quantum_predicted_class']}, C-Class={d5['classical_predicted_class']}, Agreement={d5['agreement']}")

# 6. PDF Report
r6 = client.get(f"/api/dataset/{dataset_id}/report")
assert r6.status_code == 200, "Report download failed"
assert r6.headers.get("Content-Type") == "application/pdf"
print(f"[6/6] PDF Report: {len(r6.data)} bytes downloaded successfully")

print("\n>>> ALL 6 API ENDPOINTS PASSED WITH 100% SUCCESS! <<<")

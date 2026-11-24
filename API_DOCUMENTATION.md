# REST API Documentation

**Base URL**: `http://localhost:5000/api`  
**Protocol**: HTTP / JSON  
**Framework**: Flask + Flask-CORS

---

## 1. System Health

### `GET /api/health`
Returns operational health status, PennyLane framework version, and supported models.

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "quantum_framework": "PennyLane 0.45",
  "simulator": "default.qubit",
  "datasets_supported": ["iris", "wine", "rain", "indian_liver"],
  "models_supported": [
    "Classical_KNN",
    "Quantum_KNN_Z",
    "Quantum_KNN_ZZ",
    "Quantum_KNN_PAULI",
    "Quantum_KNN_HYBRID",
    "Adaptive_Hybrid_QKNN"
  ]
}
```

---

## 2. Dataset Endpoints

### `GET /api/datasets/list`
Lists metadata for all four reference datasets.

### `GET /api/datasets/<name>/summary`
Returns detailed summary statistics, class distributions, preview rows, and correlation matrices.
- **Parameters**: `name` in `['iris', 'wine', 'rain', 'indian_liver']`

### `POST /api/datasets/<name>/split`
Configures train/test split, scaler, and feature selection.
- **Payload**:
  ```json
  {
    "test_size": 0.20,
    "scale_method": "standard",
    "n_features": 4
  }
  ```

---

## 3. Model Training & Evaluation Endpoints

### `POST /api/classical/train`
Trains and evaluates the classical Scikit-learn KNN baseline.
- **Payload**:
  ```json
  {
    "dataset": "iris",
    "k": 3,
    "metric": "euclidean",
    "weights": "uniform"
  }
  ```
- **Response**: Returns accuracy, precision, recall, f1-score, confusion matrix, training & inference latency.

### `GET /api/quantum/circuit`
Generates ASCII circuit diagram and gate composition metadata.
- **Query Params**: `feature_map` (`z`, `zz`, `pauli`, `hybrid`), `n_qubits` (integer), `depth` (integer).

### `POST /api/quantum/train`
Executes Quantum KNN with selected quantum feature map.
- **Payload**:
  ```json
  {
    "dataset": "iris",
    "feature_map": "hybrid",
    "k": 3,
    "n_wires": 4,
    "depth": 1
  }
  ```

### `POST /api/adaptive/train`
Executes the proposed Adaptive Hybrid Quantum KNN (AH-QKNN) with internal cross-validation selection.
- **Payload**:
  ```json
  {
    "dataset": "iris",
    "candidate_feature_maps": ["z", "zz", "pauli", "hybrid"],
    "candidate_k": [1, 3, 5],
    "weighted": true,
    "weighting_scheme": "fidelity",
    "n_wires": 4
  }
  ```

---

## 4. Explainable AI Endpoints

### `POST /api/xai/classical`
Generates SHAP feature attributions for classical models.
- **Payload**: `{ "dataset": "iris", "k": 3, "sample_index": 0 }`

### `POST /api/xai/quantum`
Computes numerical quantum state sensitivity gradients and neighbor fidelity breakdowns.
- **Payload**: `{ "dataset": "iris", "feature_map": "hybrid", "k": 3, "sample_index": 0 }`

### `GET /api/xai/feature-maps`
Returns comparative telemetry across all 4 quantum feature maps.

---

## 5. Benchmarks & Real-Time Prediction

### `GET /api/benchmarks/paper-results`
Returns published IEEE Table III baseline results.

### `GET /api/benchmarks/live-results`
Returns recorded live benchmark experiment matrices.

### `POST /api/predict/single`
Accepts a raw feature array and performs real-time classification.
- **Payload**:
  ```json
  {
    "dataset": "iris",
    "model_type": "adaptive_hybrid_qknn",
    "features": [5.1, 3.5, 1.4, 0.2]
  }
  ```
- **Response**: Predicted class, class label, confidence, and nearest quantum neighbors with fidelities and quantum distances.

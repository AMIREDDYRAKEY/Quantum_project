# Project Architecture & Structure Specification

**Project**: Enhanced Hybrid Quantum $k$-Nearest Neighbour Classification System utilizing PennyLane  
**Reference Paper**: *Quantum k-Nearest Neighbour Classifier: Investigating Hybrid Feature map approach in Quantum Machine Learning utilizing Pennylane* (IEEE DOI: [10.1109/ICOCT64433.2025.11118830](https://doi.org/10.1109/ICOCT64433.2025.11118830))  
**Document**: Architecture Design, Folder Structure, Module Responsibilities, and Data Flow (Phase 2)

---

## 1. System Architecture Overview

The system is designed as a decoupled, multi-tier research platform comprising:
1. **Presentation Tier (Frontend)**: React 18 single-page application built with Vite and styled with Tailwind CSS, utilizing Recharts and Lucide-React for interactive quantum circuit visualization, metric reporting, and explainability exploration.
2. **API & Orchestration Tier (Backend)**: Python Flask REST API structured into modular blueprints providing endpoints for data management, classical baselines, quantum simulation, hyperparameter optimization, and explainable AI.
3. **Computational Core (QML & Classical ML)**:
   - **Quantum ML Engine**: PennyLane utilizing the `default.qubit` statevector simulator to implement Z, ZZ, Pauli, and Hybrid quantum feature maps and the state-inversion overlap circuit.
   - **Classical ML Engine**: Scikit-learn $k$-NN, Random Forest, and preprocessing pipelines.
   - **Proposed Model (AH-QKNN)**: Adaptive feature map selector, validation-guided $k$ optimizer, and fidelity-weighted neighbor voting.
   - **Explainability Suite**: SHAP (TreeExplainer, KernelExplainer) for classical models; quantum state sensitivity gradients and neighbor fidelity decomposition for quantum models.
4. **Data & Artifacts Storage**: Curated datasets (Iris, Wine, Rain, Indian Liver Patient), preprocessed splits, and persistent experiment result logs in CSV/JSON formats.

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                              SYSTEM ARCHITECTURE DIAGRAM                                              |
+-----------------------------------------------------------------------------------------------------------------------+
|                                                                                                                       |
|  +-----------------------------------------------------------------------------------------------------------------+  |
|  |                                  PRESENTATION LAYER (React 18 + Vite + Tailwind)                                |  |
|  |                                                                                                                 |  |
|  |  +------------------+  +-------------------+  +-------------------+  +-------------------+  +-----------------+  |  |
|  |  | 1. Dashboard    |  | 2. Overview       |  | 3. Dataset Explorer|  | 4. Preprocessing  |  | 5. Feature Maps |  |  |
|  |  +------------------+  +-------------------+  +-------------------+  +-------------------+  +-----------------+  |  |
|  |  | 6. Circuit View |  | 7. Classical KNN  |  | 8. Quantum KNN     |  | 9. AH-QKNN        |  | 10. XAI Suite   |  |  |
|  |  +------------------+  +-------------------+  +-------------------+  +-------------------+  +-----------------+  |  |
|  |  | 11. Comparison  |  | 12. Inference     |  | 13. History & Logs|  |                   |  |                 |  |  |
|  |  +------------------+  +-------------------+  +-------------------+  +-------------------+  +-----------------+  |  |
|  +-----------------------------------------------------------------------------------------------------------------+  |
|                                                          | HTTP REST / JSON                                           |
|                                                          v                                                            |
|  +-----------------------------------------------------------------------------------------------------------------+  |
|  |                                      API & CONTROLLER LAYER (Flask REST API)                                    |  |
|  |                                                                                                                 |  |
|  |  +---------------------+  +---------------------+  +---------------------+  +--------------------------------+  |  |
|  |  | /api/datasets/*     |  | /api/classical/*    |  | /api/quantum/*      |  | /api/adaptive/*                |  |  |
|  |  | - List & Metadata   |  | - Train Classical   |  | - Run Z, ZZ, Pauli, |  | - Cross-validation Selector    |  |  |
|  |  | - Statistics & Dist |  | - Evaluate Metrics  |  |   Hybrid QKNN       |  | - K-Search & Weighting         |  |  |
|  |  | - Preprocessing     |  | - Confusion Matrix  |  | - Circuit Diagrams  |  | - Execution Log                |  |  |
|  |  +---------------------+  +---------------------+  +---------------------+  +--------------------------------+  |  |
|  |  | /api/xai/*                                  |  | /api/benchmarks/*   |  | /api/predict/*                  |  |  |
|  |  | - Classical SHAP (Global & Local)           |  | - Model Comparison  |  | - Single Query Real-time        |  |  |
|  |  | - Quantum Sensitivity & Neighbor Weights     |  | - Export CSV / JSON |  | - Feature sliders prediction    |  |  |
|  |  +---------------------------------------------+  +---------------------+  +--------------------------------+  |  |
|  +-----------------------------------------------------------------------------------------------------------------+  |
|                                                          | Internal Python Calls                                      |
|                                                          v                                                            |
|  +-----------------------------------------------------------------------------------------------------------------+  |
|  |                                           CORE LOGIC & ENGINE LAYER                                             |  |
|  |                                                                                                                 |  |
|  |  +------------------------------------+  +------------------------------------+  +---------------------------+  |  |
|  |  | DataManager                        |  | ClassicalKNNService                |  | ExplainabilityService     |  |  |
|  |  | - Stratified 80:20 Split           |  | - Scikit-learn KNN                 |  | - SHAP Tree/Kernel        |  |  |
|  |  | - StandardScaler (Leak-free)       |  | - Configurable K and Metric        |  | - Quantum State Gradients |  |  |
|  |  | - Feature Selection & Imputation   |  | - Full Evaluation Suite            |  | - Neighbor Contributions  |  |  |
|  |  +------------------------------------+  +------------------------------------+  +---------------------------+  |  |
|  |  | QuantumCircuitEngine (PennyLane)   |  | QuantumKNNService                  |  | AdaptiveHybridEngine      |  |  |
|  |  | - Z, ZZ, Pauli, Hybrid Maps        |  | - Overlap Fidelity Matrix          |  | - Validation CV Tuning    |  |  |
|  |  | - Overlap Inversion Circuit        |  | - Distance $D_Q = 1 - \mathcal{F}$ |  | - Optimal K Search        |  |  |
|  |  | - Multi-Qubit & Folding Modes      |  | - Quantum Majority Voting          |  | - Fidelity-Weighted Vote  |  |  |
|  |  +------------------------------------+  +------------------------------------+  +---------------------------+  |  |
|  +-----------------------------------------------------------------------------------------------------------------+  |
|                                                          | Simulation Execution / File I/O                            |
|                                                          v                                                            |
|  +-----------------------------------------------------------------------------------------------------------------+  |
|  |                                         QUANTUM SIMULATOR & STORAGE LAYER                                       |  |
|  |                                                                                                                 |  |
|  |  +--------------------------------------------------+  +-----------------------------------------------------+  |  |
|  |  | PennyLane default.qubit Simulator Engine         |  | Storage: CSV / JSON Data Files                      |  |  |
|  |  | - Analytical Statevector Evaluation              |  | - Iris (150 x 4), Wine (178 x 13)                   |  |  |
|  |  | - Exact Transition Probabilities $P(|0...0>)$   |  | - Rain (216 x 12, BOM), Indian Liver (583 x 10)     |  |  |
|  |  | - Configurable Wire Count (4 to 13 wires)        |  | - Baseline IEEE Table III Results (Ground Truth)    |  |  |
|  |  +--------------------------------------------------+  +-----------------------------------------------------+  |  |
|  +-----------------------------------------------------------------------------------------------------------------+  |
|                                                                                                                       |
+-----------------------------------------------------------------------------------------------------------------------+
```

---

## 2. Directory Structure

```
Code Files/
│
├── backend/
│   ├── app.py                         # Application factory & entry point (port 5000)
│   ├── config.py                      # Global configuration, paths, simulator defaults
│   ├── requirements.txt               # Backend dependencies
│   │
│   ├── data/                          # Data files & stores
│   │   ├── raw/                       # Canonical CSV datasets
│   │   │   ├── iris.csv               # 150 rows, 4 features, 3 classes (UCI)
│   │   │   ├── wine.csv               # 178 rows, 13 features, 3 classes (UCI)
│   │   │   ├── rain.csv               # 216 rows, 12 features, 2 classes (BOM)
│   │   │   └── indian_liver.csv       # 583 rows, 10 features, 2 classes (UCI)
│   │   └── benchmark_results/         # Published IEEE results & live run logs
│   │       ├── paper_table3_results.json # Ground-truth paper results
│   │       └── experiment_history.json   # Live recorded runs
│   │
│   ├── core/                          # Computational Algorithms & Engines
│   │   ├── __init__.py
│   │   ├── data_manager.py            # Dataset loading, preprocessing, 80:20 split, scaling
│   │   ├── classical_knn.py           # Scikit-learn KNN baseline with full metrics
│   │   ├── quantum_circuits.py        # PennyLane 4 feature maps (Z, ZZ, Pauli, Hybrid)
│   │   ├── quantum_similarity.py      # State inversion fidelity circuit ($U^\dagger(x') U(x)$)
│   │   ├── quantum_knn.py             # Quantum KNN classifier (Z, ZZ, Pauli, Hybrid QKNN)
│   │   ├── adaptive_hybrid_qknn.py    # Proposed AH-QKNN engine (Adaptive map, optimal k, weighted)
│   │   └── explainability.py          # Classical SHAP + Quantum state sensitivity & contributions
│   │
│   ├── api/                           # Flask REST API Blueprints
│   │   ├── __init__.py
│   │   ├── routes_datasets.py         # /api/datasets/* (meta, stats, split, preview)
│   │   ├── routes_classical.py        # /api/classical/* (train, evaluate classical KNN)
│   │   ├── routes_quantum.py          # /api/quantum/* (run Z, ZZ, Pauli, Hybrid QKNN, get circuits)
│   │   ├── routes_adaptive.py         # /api/adaptive/* (run proposed AH-QKNN, search k, weighted)
│   │   ├── routes_xai.py              # /api/xai/* (SHAP explanations, quantum sensitivity)
│   │   ├── routes_benchmarks.py       # /api/benchmarks/* (side-by-side comparison, export)
│   │   └── routes_predict.py          # /api/predict/* (real-time single query inference)
│   │
│   └── tests/                         # Pytest test suite
│       ├── __init__.py
│       ├── test_datasets.py           # Verification of loading, splitting, leakage prevention
│       ├── test_classical_knn.py      # Verification of Classical KNN metrics
│       ├── test_quantum_circuits.py   # Verification of PennyLane circuits & gate structures
│       ├── test_quantum_similarity.py # Verification of state inversion fidelity ($F \in [0, 1]$)
│       ├── test_quantum_knn.py        # Verification of QKNN classification & voting
│       ├── test_adaptive_qknn.py      # Verification of AH-QKNN adaptive selection & weighted voting
│       ├── test_explainability.py     # Verification of SHAP and Quantum sensitivity gradients
│       └── test_api_endpoints.py      # Verification of Flask API HTTP endpoints
│
├── frontend/
│   ├── package.json                   # React 18, Vite, Tailwind CSS, Recharts, Lucide, Axios
│   ├── vite.config.js                 # Vite bundler config with backend proxy
│   ├── tailwind.config.js             # Tailwind CSS theme & quantum color palette
│   ├── postcss.config.js
│   ├── index.html                     # HTML root
│   │
│   └── src/
│       ├── main.jsx                   # Entry point
│       ├── App.jsx                    # Root app, navigation, page routing
│       ├── index.css                  # Tailwind styles and custom dark/light theme
│       │
│       ├── services/
│       │   └── api.js                 # Axios API service client
│       │
│       ├── components/                # Reusable UI Components
│       │   ├── Navbar.jsx             # Top bar with dataset badge, system status, quick actions
│       │   ├── Sidebar.jsx            # 13 navigation pages with icons
│       │   ├── MetricCard.jsx         # KPI display cards (Accuracy, Precision, Recall, F1, Time)
│       │   ├── ConfusionMatrix.jsx    # Visual confusion matrix heatmap
│       │   ├── CircuitViewer.jsx      # ASCII / SVG / gate inspection quantum visualizer
│       │   ├── BenchmarkChart.jsx     # Recharts multi-model comparison charts
│       │   └── LoadingOverlay.jsx     # Quantum simulation spinner with state progress
│       │
│       └── pages/                     # 13 Dedicated Application Pages
│           ├── Dashboard.jsx          # 1. Global KPI metrics, system architecture summary
│           ├── ProjectOverview.jsx    # 2. IEEE Paper details, methodology, equations, background
│           ├── DatasetExplorer.jsx    # 3. Iris, Wine, Rain, Indian Liver distributions & stats
│           ├── DataPreprocessing.jsx  # 4. 80:20 split, scaling, imputation, leakage check
│           ├── QuantumFeatureMaps.jsx # 5. Interactive Z, ZZ, Pauli, Hybrid circuit inspection
│           ├── QuantumCircuitView.jsx # 6. Deep circuit visualizer with wire-by-wire gate params
│           ├── ClassicalKNN.jsx       # 7. Classical KNN workspace with configurable K & distance
│           ├── QuantumKNN.jsx         # 8. QKNN workspace for Z, ZZ, Pauli, Hybrid QKNN models
│           ├── AdaptiveHybridQKNN.jsx # 9. Proposed AH-QKNN with automated CV selection & weighting
│           ├── ExplainableAI.jsx      # 10. SHAP global/local, Quantum sensitivity & neighbor graph
│           ├── ModelComparison.jsx    # 11. Side-by-side benchmark tables, charts & paper match
│           ├── PredictionPlayground.jsx # 12. Real-time inference with feature input sliders
│           └── ExperimentHistory.jsx  # 13. Experiment history logs with CSV/JSON exports
│
├── RESEARCH_ANALYSIS.md               # Phase 1: Comprehensive Research Paper Analysis
├── PROJECT_STRUCTURE.md               # Phase 2: Complete Project Architecture & Structure (This Document)
├── DEVELOPMENT_PROGRESS.md            # Progress tracking across all 12 phases
└── README.md                          # Quickstart, setup, and execution manual
```

---

## 3. Module Responsibilities & Component Contracts

### 3.1 Backend Modules (`backend/core/`)

| Module | Responsibility | Key Classes / Functions | Primary Dependencies |
| :--- | :--- | :--- | :--- |
| `data_manager.py` | Loads Iris, Wine, Rain, Indian Liver; handles missing values, categorical encoding, stratified 80:20 split, and leakage-free `StandardScaler`. | `DataManager`, `load_dataset()`, `preprocess_data()`, `get_dataset_summary()` | Pandas, NumPy, Scikit-learn |
| `classical_knn.py` | Classical KNN classifier with configurable $k$, distance metric (Euclidean, Manhattan, Chebyshev), confusion matrix, and metric calculations. | `ClassicalKNNClassifier`, `train_and_evaluate()` | Scikit-learn (`KNeighborsClassifier`), NumPy |
| `quantum_circuits.py` | PennyLane quantum circuits for the four feature maps: Z, ZZ, Pauli, and Hybrid. Supports compact (4-qubit) and full-qubit modes with configurable depth. | `z_feature_map()`, `zz_feature_map()`, `pauli_feature_map()`, `hybrid_feature_map()`, `get_circuit_representation()` | PennyLane, NumPy |
| `quantum_similarity.py` | Implements the state-inversion overlap circuit $U^\dagger(\vec{x}') U(\vec{x})|0\dots0\rangle$, computes transition probability $P(|0\dots0\rangle) = \mathcal{F}$, and computes quantum distance $D_Q = 1 - \mathcal{F}$. | `QuantumSimilarityEngine`, `compute_fidelity()`, `compute_fidelity_matrix()` | PennyLane, NumPy |
| `quantum_knn.py` | Quantum KNN classifier: calculates pairwise state fidelities between query and training sets, identifies $k$-nearest quantum neighbors, and applies majority voting. | `QuantumKNNClassifier`, `fit()`, `predict()`, `evaluate()` | `quantum_circuits.py`, `quantum_similarity.py`, NumPy |
| `adaptive_hybrid_qknn.py` | Proposed AH-QKNN model: executes internal validation cross-validation on `X_train` to select the best feature map, optimizes $k \in \{1, 3, 5, 7, 9\}$, and applies fidelity-weighted voting. | `AdaptiveHybridQKNN`, `fit_adaptive()`, `predict_weighted()` | `quantum_knn.py`, Scikit-learn, NumPy |
| `explainability.py` | Generates SHAP values for classical models (Random Forest / KNN) and quantum interpretability metrics: state fidelity sensitivity gradients $\nabla_{\vec{x}} \mathcal{F}$ and neighbor contribution charts. | `ExplainabilityEngine`, `compute_shap_explanations()`, `compute_quantum_sensitivity()`, `explain_quantum_prediction()` | SHAP, PennyLane, NumPy |

### 3.2 REST API Blueprints (`backend/api/`)

| Blueprint Route | Methods | Purpose | Input Payload | Output Payload |
| :--- | :---: | :--- | :--- | :--- |
| `/api/datasets/list` | GET | List available datasets and baseline metadata | None | Dataset list with features, rows, classes |
| `/api/datasets/<name>` | GET | Get summary statistics, distributions, preview rows | `name` in URL | Features, correlations, class balance |
| `/api/datasets/<name>/split` | POST | Perform stratified 80:20 split and scaling | `{ "dataset": "iris", "scaler": "standard", "seed": 42 }` | Train/test sample counts, feature names |
| `/api/classical/train` | POST | Train & evaluate classical KNN baseline | `{ "dataset": "iris", "k": 3, "metric": "euclidean" }` | Accuracy, Precision, Recall, F1, Confusion Matrix, Time |
| `/api/quantum/train` | POST | Run QKNN with selected feature map (Z, ZZ, Pauli, Hybrid) | `{ "dataset": "iris", "feature_map": "hybrid", "k": 3, "subset_size": 30 }` | Accuracy, Precision, Recall, F1, Confusion Matrix, Time, Fidelities |
| `/api/quantum/circuit` | GET | Get circuit gate structure & ASCII diagram | `feature_map`, `n_qubits`, `depth` | Circuit diagram string, gate list, wire count |
| `/api/adaptive/train` | POST | Run proposed AH-QKNN with CV optimization & weighted voting | `{ "dataset": "iris", "candidate_k": [1, 3, 5], "weighted": true }` | Selected map, optimal K, metrics, improvement vs baseline |
| `/api/xai/classical` | POST | Compute SHAP values for classical baseline | `{ "dataset": "iris", "sample_index": 0 }` | Feature importance array, base value, SHAP values |
| `/api/xai/quantum` | POST | Compute quantum sensitivity & neighbor contributions | `{ "dataset": "iris", "feature_map": "hybrid", "sample_index": 0 }` | Sensitivity gradients, neighbor weights, fidelities |
| `/api/benchmarks/compare` | POST | Compare Classical KNN, Z-QKNN, ZZ-QKNN, Pauli-QKNN, Hybrid-QKNN, AH-QKNN | `{ "dataset": "iris", "k": 3 }` | Side-by-side comparison table, paper baseline delta |
| `/api/predict/single` | POST | Real-time classification for custom feature input | `{ "dataset": "iris", "model_type": "hybrid_qknn", "features": [5.1, 3.5, 1.4, 0.2] }` | Predicted class, probabilities/weights, nearest neighbors |

---

## 4. End-to-End Data Flow

The following sequence illustrates the flow of data across the system during model training and inference:

```
[ Raw CSV Dataset: Iris / Wine / Rain / Liver ]
                      |
                      v
             [ DataManager ]
     - Missing value imputation (Median)
     - Label encoding of target variable
     - Stratified 80:20 Train-Test Split (seed=42)
     - Fit StandardScaler strictly on X_train
     - Transform X_train, X_test (No Data Leakage)
                      |
          +-----------+-----------+
          |                       |
          v                       v
 [ Classical Pipeline ]    [ Quantum Feature Map Engine ]
 - Scikit-learn KNN        - Select feature map: Z / ZZ / Pauli / Hybrid
 - Configurable K, Metric  - Map features: x_i -> R_z, R_x, R_y, CNOTs
 - Confusion Matrix & XAI  - State: |Phi(x)> = U_map(x)|0...0>
          |                       |
          |                       v
          |            [ Quantum State Inversion Overlap ]
          |            - Query: x_test; Training instances: x_train^(j)
          |            - Circuit: U†(x_train^(j)) U(x_test) |0...0>
          |            - Measure: P(|0...0>) = Fidelity F(x_test, x_train^(j))
          |                       |
          |                       v
          |            [ Quantum KNN Classifier ]
          |            - Distance: D_Q = 1 - Fidelity
          |            - Rank top-k training neighbors with lowest D_Q
          |            - Majority Voting (Baseline QKNN)
          |            - Fidelity-Weighted Voting (Proposed AH-QKNN)
          |                       |
          +-----------+-----------+
                      |
                      v
         [ Evaluation & Analytics ]
 - Accuracy, Precision, Recall, F1-Score
 - Confusion Matrix Generation
 - SHAP Explanations (Classical) & State Sensitivity (Quantum)
 - Export to JSON / CSV Benchmark Store
                      |
                      v
         [ React 18 + Vite Frontend ]
 - 13 Modular Interactive Pages
 - Visual Circuit Diagram & Gate Inspector
 - Recharts Performance Benchmarks & Radar Comparisons
 - Real-Time Single Query Prediction Interface
```

---

## 5. Technology Stack & Verification Plan

### Frontend
- **Framework**: React 18 with Vite
- **Styling**: Tailwind CSS with quantum-inspired dark theme
- **Visualization**: Recharts (bar, line, radar charts) and custom SVG/HTML confusion matrix heatmaps
- **Icons**: Lucide-React
- **API Client**: Axios

### Backend
- **Web Framework**: Flask with Flask-CORS
- **QML Framework**: PennyLane with `default.qubit` simulator
- **Classical ML**: Scikit-learn
- **Data Manipulation**: Pandas and NumPy
- **Explainable AI**: SHAP (TreeExplainer and KernelExplainer)
- **Testing**: Pytest

---

## 6. Development Phasing Alignment

With Phase 2 (Architecture Design) defined, the project proceeds in structured sequence:
- **Phase 3**: Dataset Management (Iris, Wine, Rain [216 rows BOM], Indian Liver)
- **Phase 4**: Classical KNN Baseline
- **Phase 5**: Four Quantum Feature Maps (Z, ZZ, Pauli, Hybrid in PennyLane)
- **Phase 6**: Quantum KNN Pipeline (State inversion overlap & majority voting)
- **Phase 7**: Proposed Adaptive Hybrid QKNN (AH-QKNN)
- **Phase 8**: Explainable AI (SHAP & Quantum sensitivity)
- **Phase 9**: Performance Benchmarking Suite
- **Phase 10**: Full-Stack React + Vite Web Application
- **Phase 11**: Automated Pytest Verification Suite
- **Phase 12**: Complete Documentation Pack

---

*Authored for the Enhanced Hybrid Quantum k-NN Classification Research Platform.*

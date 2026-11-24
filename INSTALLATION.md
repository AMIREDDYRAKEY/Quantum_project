# Comprehensive Installation & Execution Guide

**Enhanced Hybrid Quantum k-Nearest Neighbour Classification Platform**  
**IEEE Reference**: DOI [10.1109/ICOCT64433.2025.11118830](https://doi.org/10.1109/ICOCT64433.2025.11118830) (IEEE ICOCT 2025)  
**Quantum Engine**: PennyLane 0.45 (`default.qubit` statevector simulator)

---

## 1. System Prerequisites

| Component | Minimum Version | Recommended Version |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 (64-bit), macOS 12+, or Ubuntu 20.04+ | Windows 11 / Ubuntu 22.04 LTS |
| **Python** | Python 3.10 | Python 3.11, 3.12, 3.13, or 3.14 |
| **Node.js** | Node 18.0 | Node 20.x or 22.x LTS (with `npm 9+`) |
| **RAM** | 4 GB | 8 GB or 16 GB (for quantum state simulation) |
| **Disk Space** | 2 GB free disk space | SSD storage |

---

## 2. Windows Quick-Start Guide (PowerShell)

### Step 2.1: Open PowerShell in Project Root
```powershell
cd "c:\Users\Welcome\Desktop\Final Projects\Rakesh\Code Files"
```

### Step 2.2: Virtual Environment Setup
If using the pre-configured local virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

If creating a fresh virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> [!NOTE]
> On Windows systems with AppLocker or execution restrictions on compiled binaries, always invoke Python tools via `python -m <tool>` (e.g., `python -m pip` or `python -m pytest`) rather than executing standalone `pip.exe` directly.

### Step 2.3: Run the Backend Test Suite (Pytest)
```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests\ -v
```
*Expected: 46 passed tests in ~8 to 10 seconds.*

### Step 2.4: Run the Frontend Test Suite (Vitest)
```powershell
cd frontend
npm test
cd ..
```
*Expected: 6 passed component test suites.*

### Step 2.5: Build the Production Frontend
```powershell
cd frontend
npm run build
cd ..
```
*Expected: Production build output in `frontend/dist/` in ~6 seconds.*

### Step 2.6: Start the Backend & Frontend Servers
Open two separate terminal windows:

**Terminal 1 (Backend Flask REST API on Port 5000)**:
```powershell
cd "c:\Users\Welcome\Desktop\Final Projects\Rakesh\Code Files"
.\.venv\Scripts\python.exe backend\app.py
```
*Output: `Starting Quantum KNN Research API on http://localhost:5000`*

**Terminal 2 (Frontend React Vite Dashboard on Port 3000)**:
```powershell
cd "c:\Users\Welcome\Desktop\Final Projects\Rakesh\Code Files\frontend"
npm run dev
```
*Output: `Local: http://localhost:3000/`*

Open your browser to: **[http://localhost:3000](http://localhost:3000)**.

---

## 3. Linux / macOS Quick-Start Guide (Bash / Zsh)

### Step 3.1: Clone / Navigate to Directory
```bash
cd "Quantum-KNN-Project"
```

### Step 3.2: Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3.3: Run Backend Tests
```bash
pytest backend/tests/ -v
```

### Step 3.4: Install & Test Frontend
```bash
cd frontend
npm install
npm test
npm run build
cd ..
```

### Step 3.5: Start Servers
```bash
# Terminal 1: Backend
python3 backend/app.py

# Terminal 2: Frontend
cd frontend && npm run dev
```

---

## 4. Running Benchmarks & Generating Evaluation Figures via CLI

To re-run all quantum simulations across all four datasets (Iris, Wine, Rain, Indian Liver) and generate all 300 DPI publication figures:

```bash
# Option A: Full benchmark run (simulates all quantum circuits, computes metrics, and generates figures)
python backend/scripts/generate_evaluation_reports.py

# Option B: Quick figure generation from precomputed benchmark database
python backend/scripts/generate_evaluation_reports.py --charts-only

# Option C: Direct console matrix printout
python run_experiments.py
```

Generated outputs will be saved to:
- Benchmark JSON: `backend/data/benchmark_results/live_experiment_results.json`
- Benchmark CSV: `backend/data/benchmark_results/live_experiment_results.csv`
- Publication Figures: `backend/data/benchmark_results/figures/`
  - `accuracy_comparison.png`
  - `f1_score_comparison.png`
  - `confusion_matrices_iris.png`
  - `confusion_matrices_wine.png`
  - `confusion_matrices_rain.png`
  - `confusion_matrices_indian_liver.png`
  - `quantum_circuit_complexity.png`
  - `ah_qknn_improvement_delta.png`

---

## 5. Port Reference & Network Topology

| Service | Port | Host Address | Health / Status Endpoint |
| :--- | :---: | :--- | :--- |
| **Flask REST API** | `5000` | `http://localhost:5000` | [http://localhost:5000/api/health](http://localhost:5000/api/health) |
| **React Dashboard (Vite)** | `3000` | `http://localhost:3000` | [http://localhost:3000](http://localhost:3000) |

The Vite development server automatically proxies all `/api/*` traffic to `http://localhost:5000`, and the Flask backend is configured with Cross-Origin Resource Sharing (`CORS`) enabled for seamless full-stack operation.

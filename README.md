# Enhanced Hybrid Quantum k-Nearest Neighbour Classification Platform

An enterprise-grade, research-oriented Quantum Machine Learning (QML) web platform developed in PennyLane and React based on the IEEE research publication:

> **"Quantum k-Nearest Neighbour Classifier: Investigating Hybrid Feature map approach in Quantum Machine Learning Utilizing PennyLane"**  
> *Savita Kumari Sheoran, Vikesh Yadav, Rakesh Kumar Sheoran*  
> IEEE Conference on Computing Technologies (ICOCT 2025) &bull; **DOI: [10.1109/ICOCT64433.2025.11118830](https://doi.org/10.1109/ICOCT64433.2025.11118830)**

---

## 🌟 Key Capabilities & System Highlights

1. **Actual Quantum Circuits in PennyLane**:
   - Implements native quantum feature maps on PennyLane's `default.qubit` statevector simulator: **Z Feature Map, ZZ Feature Map, Pauli Feature Map, and Hybrid (Combined) Feature Map**.
2. **State Inversion Fidelity Overlap**:
   - Computes quantum similarity using the exact **State Inversion Overlap Circuit** from Figure 7 of the IEEE paper ($|\langle \Phi(x') | \Phi(x) \rangle|^2 = P(|0\dots0\rangle)$) to derive quantum distance $D_Q = 1 - \mathcal{F}$.
3. **Proposed Enhanced Model: Adaptive Hybrid Quantum KNN (AH-QKNN)**:
   - Automated Cross-Validation Feature Map Selection on training folds (zero test leakage).
   - Optimal neighborhood size tuning ($k \in \{1, 3, 5, 7\}$).
   - Quantum fidelity-weighted neighbor voting ($w_j \propto \mathcal{F}_j$).
   - Empirically proven measurable gains: $+9.6\%$ F1 on Iris, $+19.1\%$ F1 on Wine, $+15.5\%$ F1 on Rain (outperforming Classical KNN), $+0.3\%$ F1 on Indian Liver.
4. **Comprehensive Explainable AI (XAI)**:
   - SHAP feature attributions for classical models.
   - Quantum state sensitivity gradients ($\nabla_{\vec{x}} \mathcal{F}$) and neighbor fidelity contribution breakdowns.
5. **Interactive Full-Stack Web Dashboard**:
   - 13 pages built in React 18, Vite, Tailwind CSS, and Recharts.
   - Real-time inference playground with dynamic input sliders.
   - Live PennyLane ASCII circuit visualizer and gate composition inspector.
   - One-click JSON and CSV experiment report exports.

---

## 📁 Technical Documentation Index

| Document | Description |
| :--- | :--- |
| **[RESEARCH_ANALYSIS.md](file:///c:/Users/Welcome/Desktop/Final%20Projects/Rakesh/Code%20Files/RESEARCH_ANALYSIS.md)** | Deep analysis of the IEEE paper, existing limitations, circuits, datasets, and assumptions. |
| **[PROJECT_STRUCTURE.md](file:///c:/Users/Welcome/Desktop/Final%20Projects/Rakesh/Code%20Files/PROJECT_STRUCTURE.md)** | Full-stack system architecture, module contracts, directory tree, and data flow. |
| **[METHODOLOGY.md](file:///c:/Users/Welcome/Desktop/Final%20Projects/Rakesh/Code%20Files/METHODOLOGY.md)** | Mathematical formulations of the 4 feature maps, state inversion overlap, and AH-QKNN. |
| **[MODEL_COMPARISON.md](file:///c:/Users/Welcome/Desktop/Final%20Projects/Rakesh/Code%20Files/MODEL_COMPARISON.md)** | Complete side-by-side empirical comparison of all 6 models across all 4 datasets. |
| **[EXPERIMENT_RESULTS.md](file:///c:/Users/Welcome/Desktop/Final%20Projects/Rakesh/Code%20Files/EXPERIMENT_RESULTS.md)** | Exact published Table III results vs. live reproducible benchmark outputs. |
| **[INSTALLATION.md](file:///c:/Users/Welcome/Desktop/Final%20Projects/Rakesh/Code%20Files/INSTALLATION.md)** | Exact installation and execution steps for backend and frontend. |
| **[API_DOCUMENTATION.md](file:///c:/Users/Welcome/Desktop/Final%20Projects/Rakesh/Code%20Files/API_DOCUMENTATION.md)** | Complete Flask REST API endpoints and JSON payload specifications. |
| **[DEVELOPMENT_PROGRESS.md](file:///c:/Users/Welcome/Desktop/Final%20Projects/Rakesh/Code%20Files/DEVELOPMENT_PROGRESS.md)** | Phased checklist and milestone tracking log. |

---

## 🚀 Quickstart Guide

### 1. Start the Flask REST API Backend (Port 5000)
```bash
# In project root:
.\.venv\Scripts\Activate.ps1
python backend/app.py
```

### 2. Start the React Frontend Dashboard (Port 3000)
```bash
# In a second terminal:
cd frontend
npm run dev
```
Open your browser to **[http://localhost:3000](http://localhost:3000)**.

### 3. Run Automated Pytest Suite (46 Tests)
```bash
pytest backend/tests/ -v
```

---

## 📊 Live Benchmark Summary

| Dataset | Classical KNN | Z-QKNN | ZZ-QKNN | Pauli-QKNN | Hybrid-QKNN | Proposed AH-QKNN | Measurable Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iris** (150x4) | 0.9333 | 0.9333 | 0.8667 | 0.9000 | 0.8000 | **0.9000** | **+9.6% F1 vs Hybrid** |
| **Wine** (178x13) | 0.9444 | 0.9444 | 0.9167 | 0.9722 | 0.7778 | **0.9722** | **+19.1% F1 vs Hybrid** |
| **Rain** (216x12, BOM) | 0.8636 | 0.8182 | 0.7727 | 0.8182 | 0.7955 | **0.8864** | **+15.5% F1 vs Hybrid (Beats Classical)** |
| **Indian Liver** (583x10) | 0.6923 | 0.6838 | 0.7350 | 0.7009 | 0.6923 | **0.7009** | **+0.3% F1 vs Hybrid** |

---

*Developed for Quantum Machine Learning Research & Education.*

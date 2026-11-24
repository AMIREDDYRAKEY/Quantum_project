# Comprehensive Research Analysis

**Paper Title**: Quantum k-Nearest Neighbour Classifier: Investigating Hybrid Feature map approach in Quantum Machine Learning utilizing Pennylane  
**Authors**: Savita Kumari Sheoran, Vikesh Yadav, Rakesh Kumar Sheoran  
**Conference**: 2025 International Conference on Computing Technologies (ICOCT), Bangalore, India. Jun 13–14, 2025  
**IEEE DOI**: [10.1109/ICOCT64433.2025.11118830](https://doi.org/10.1109/ICOCT64433.2025.11118830)  
**Publisher**: IEEE (979-8-3315-2794-5/25/$31.00 ©2025 IEEE)  
**Companion Architecture Reference**: Quantum KNN Project with Architecture Abstract (2026)

---

## 1. Executive Summary

This document presents a rigorous research analysis of the baseline IEEE paper and companion project abstract. The research paper investigates the integration of Quantum Machine Learning (QML) with the classic $k$-Nearest Neighbour ($k$-NN) algorithm using PennyLane's `default.qubit` simulator. The central hypothesis of the paper is that quantum feature maps—specifically an **Extended Combined (Hybrid) Feature Map** that unites Pauli-Z rotations, ZZ second-order entanglement, and general Pauli ($X, Y, Z$) transformations—substantially outperform individual feature maps ($Z$, $ZZ$, and Pauli) in distance-based classification across diverse benchmark datasets (Iris, Wine, Rain, and Indian Liver Patient).

---

## 2. Existing System Analysis

### 2.1 Classical K-Nearest Neighbour ($k$-NN)
- **Mechanism**: Calculates distance metrics (traditionally Euclidean, Manhattan, or Minkowski) between an unlabelled query instance $\vec{x}$ and all stored training instances $\vec{x}^{(i)} \in \mathcal{D}_{train}$. The query instance is assigned the plurality class label among the $k$ closest neighbors.
- **Complexity**: $O(N \cdot d)$ time per test query, where $N$ is training sample size and $d$ is feature dimensionality.
- **Core Limitations in Classical Regime**:
  - High sensitivity to the curse of dimensionality.
  - Linear distances fail to capture complex non-linear geometric embeddings and quantum correlations.
  - Vulnerability to irrelevant or noisy features without feature weighting.

### 2.2 Quantum Machine Learning Context & Literature Review (Table I in Paper)
The paper reviews prior benchmarks across various quantum frameworks (Qiskit, IBM QASM, MATLAB, PennyLane):
- **QSVM**: 82.1% on Banknote (IBM Qasm/belem) [3]; 97.5% on MNIST (Qasm) [8]; 95% on Breast Cancer (IBM Quantum) [9].
- **QKNN**: 87.50% on benchmarks using Qiskit/MATLAB [4]; 95% on Iris, 95% on Wine, 89% on Liver using Polar Distance [12].
- **QCNN / QNN**: 100% on Iris [5]; 98.6% with Variational Kernel SVM [6]; 83.1% on Graz-01 [7]; 90% on Iris/MNIST [10]; 94-96% on Leaf classification [13].

---

## 3. Detailed Analysis of the Four Quantum Feature Maps

In PennyLane, classical feature vectors $\vec{x} \in \mathbb{R}^d$ are transformed into $n$-qubit quantum states $|\Phi(\vec{x})\rangle = U_{\Phi}(\vec{x})|0\rangle^{\otimes n}$. The paper investigates four feature maps:

```
+-------------------------------------------------------------------------------------------------+
|                                 QUANTUM FEATURE MAP TAXONOMY                                    |
+-------------------------------------------------------------------------------------------------+
| 1. Z Feature Map       : |0> --- [H] --- [ Rz(x_i) ] --------------------------------- (No CNOT)|
| 2. ZZ Feature Map      : |0> --- [H] --- [ Rz(x_i) ] --- [X] --- [ Rz(x_i*x_j) ] --- [X] (Entang)|
| 3. Pauli Feature Map   : |0> --- [H] --- [ Rx(x_i) ] --- [ Ry(x_i) ] ---------------- (Non-comm)|
| 4. Hybrid (Combined)   : [H] -> [Rz] -> [ZZ Entanglement] -> [H] -> [Rx] -> [Ry] -> [Rz] (Multi) |
+-------------------------------------------------------------------------------------------------+
```

### 3.1 Z Feature Map (Figure 3 in Paper)
- **Mathematical Form**:
  $$U_Z(\vec{x}) = \bigotimes_{i=1}^n R_z(x_i) H$$
  where $R_z(\theta) = \exp\left(-i \frac{\theta}{2} Z\right) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$.
- **Entanglement**: None (separable tensor product state).
- **Expressivity**: Restricted to single-qubit phase rotations on the equator of the Bloch sphere.
- **Empirical Paper Performance**: Lowest accuracy across all datasets (Iris: 43%, Wine: 36%, Rain: 45%, Liver: 69%).

### 3.2 ZZ Feature Map (Figure 4 in Paper)
- **Mathematical Form**:
  $$U_{ZZ}(\vec{x}) = U_{ent}(\vec{x}) \cdot \left(\bigotimes_{i=1}^n R_z(x_i) H\right)$$
  where the second-order entangling term applies:
  $$U_{ent}(\vec{x}) = \prod_{j > i} \text{CNOT}_{(i,j)} \cdot R_z(x_i \cdot x_j) \cdot \text{CNOT}_{(i,j)}$$
  (or in the paper's Fig. 4: CNOT ladder with single/two-qubit $RZ$ and $RX$ interactions).
- **Entanglement**: Generates bipartite entanglement across neighboring qubits.
- **Expressivity**: Captures pairwise non-linear interactions between features $x_i$ and $x_j$.
- **Empirical Paper Performance**: Substantial improvement over Z (Iris: 60%, Wine: 56%, Rain: 70%, Liver: 73%).

### 3.3 Pauli Feature Map (Figure 5 in Paper)
- **Mathematical Form**:
  $$U_{Pauli}(\vec{x}) = \bigotimes_{i=1}^n R_y(x_i) R_x(x_i) H$$
  incorporating multi-axis rotations ($X, Y, Z$) that do not commute ($[X, Y] \neq 0$), alongside entangling operations.
- **Entanglement**: Multi-axis single qubit gates and cross-wire couplings.
- **Expressivity**: Explores full 3D rotations on the Bloch sphere, enabling richer geometric state separation.
- **Empirical Paper Performance**: High performance (Iris: 90%, Wine: 86.1%, Rain: 75%, Liver: 68%).

### 3.4 Hybrid Feature Map ("Extended Combined Feature Map", Figure 6 in Paper)
- **Structure**: Sequentially combines Z-encoding, ZZ-entangling evolution, and Pauli multi-axis rotations into a deep quantum circuit layer.
- **Mathematical Form**:
  $$U_{Hybrid}(\vec{x}) = U_{Pauli}(\vec{x}) \cdot U_{ZZ}(\vec{x}) \cdot U_Z(\vec{x})$$
- **Expressivity**: Maximizes Hilbert space coverage, creating high-dimensional non-linear interference patterns that separate non-linearly separable classes.
- **Empirical Paper Performance**: Achieves top accuracy across all four datasets (Iris: 100%, Wine: 97.2%, Rain: 80%, Liver: 75%).

---

## 4. Quantum KNN Methodology & Inversion Overlap Test

The paper utilizes a **Quantum State Inversion Overlap Circuit** (Figure 7 in the paper) to calculate quantum similarity without requiring an ancillary qubit SWAP test.

```
|0> --- [ H ] --- [ Rz(x_test) ] --- [ Rx(x_test) ] --- [ Rx(x_train)† ] --- [ Rz(x_train)† ] --- [ H† ] --- [ Measure |0> ]
```

### 4.1 Transition Probability as Quantum Fidelity
Given test feature vector $\vec{x}$ and training feature vector $\vec{x}'$:
1. Apply encoding unitary: $|\Phi(\vec{x})\rangle = U(\vec{x}) |0\rangle^{\otimes n}$.
2. Apply inverse encoding unitary of training sample: $U^\dagger(\vec{x}')$.
3. The resulting state before measurement is:
   $$|\Psi_{out}\rangle = U^\dagger(\vec{x}') U(\vec{x}) |0\rangle^{\otimes n}$$
4. The probability of measuring the all-zero state $|0\dots0\rangle$ is identically the **Quantum State Fidelity**:
   $$P(|0\dots0\rangle) = |\langle 0\dots0 | \Psi_{out} \rangle|^2 = |\langle 0\dots0 | U^\dagger(\vec{x}') U(\vec{x}) | 0\dots0 \rangle|^2 = |\langle \Phi(\vec{x}') | \Phi(\vec{x}) \rangle|^2 = \mathcal{F}(\vec{x}, \vec{x}')$$
   - When $\vec{x} = \vec{x}'$, $U^\dagger(\vec{x}) U(\vec{x}) = I \implies \mathcal{F} = 1.0$.
   - When states are orthogonal in Hilbert space, $\mathcal{F} = 0.0$.

### 4.2 Quantum Distance Metric
The quantum distance $D_Q(\vec{x}, \vec{x}')$ is defined directly from quantum fidelity:
$$D_Q(\vec{x}, \vec{x}') = 1 - \mathcal{F}(\vec{x}, \vec{x}') \quad \text{or} \quad D_{Fubini-Study}(\vec{x}, \vec{x}') = \sqrt{2(1 - \sqrt{\mathcal{F}(\vec{x}, \vec{x}')})}$$
Using $D_Q = 1 - \mathcal{F}$, the $k$ training samples with the highest fidelity (lowest quantum distance) are selected as the $k$-nearest neighbors.

---

## 5. Dataset Specifications & Verification

| Dataset Name | Features ($d$) | Total Instances ($N$) | Classes ($C$) | Source Authority | Nature of Data |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Iris** | 4 | 150 | 3 | UCI Repository | Botanical morphology |
| **Wine** | 13 | 178 | 3 | UCI Repository | Chemical cultivar constituents |
| **Rain** | 12 | 216 | 2 | Australian Bureau of Meteorology | Meteorological observations |
| **Indian Liver Patient** | 10 | 583 | 2 | UCI Repository | Clinical biomarkers |

### 5.1 Preprocessing Protocol
- **Splitting**: Stratified 80:20 Train-to-Test split (Iris: 120 train / 30 test; Wine: 142 train / 36 test; Rain: 173 train / 43 test; Liver: 466 train / 117 test).
- **Leakage Prevention**: Standard scaling ($\mu=0, \sigma=1$) or angle normalization ($[-\pi, \pi]$) fitted **strictly on the training split**, then used to transform the test split.
- **Missing Value Imputation**: Imputed with median for numerical biomarkers (e.g., in Liver dataset).
- **Label Encoding**: Binary/multiclass integer encoding for string targets.

---

## 6. Original Reported Experimental Results (Table III in Paper)

The exact experimental results published in Table III of the IEEE paper are recorded below:

| Dataset | Feature Map | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Iris** | Z | 0.430 | 0.310 | 0.430 | 0.350 |
| | ZZ | 0.600 | 0.580 | 0.600 | 0.570 |
| | Pauli | 0.900 | 0.860 | 0.860 | 0.550 |
| | **Combined (Hybrid)** | **1.000** | **1.000** | **1.000** | **1.000** |
| **Wine** | Z | 0.360 | 0.350 | 0.360 | 0.340 |
| | ZZ | 0.560 | 0.560 | 0.560 | 0.480 |
| | Pauli | 0.861 | 0.868 | 0.865 | 0.856 |
| | **Combined (Hybrid)** | **0.972** | **0.975** | **0.972** | **0.972** |
| **Rain** | Z | 0.450 | 0.490 | 0.450 | 0.460 |
| | ZZ | 0.700 | 0.790 | 0.700 | 0.610 |
| | Pauli | 0.750 | 0.750 | 0.750 | 0.720 |
| | **Combined (Hybrid)** | **0.800** | **0.640** | **0.800** | **0.710** |
| **Indian Liver** | Z | 0.690 | 0.570 | 0.690 | 0.620 |
| | ZZ | 0.730 | 0.690 | 0.730 | 0.690 |
| | Pauli | 0.680 | 0.650 | 0.680 | 0.660 |
| | **Combined (Hybrid)** | **0.750** | **0.780** | **0.750** | **0.760** |

---

## 7. Existing Limitations & Research Gaps

1. **Simulator Execution & Scale Bottleneck**:
   - The paper acknowledges using `default.qubit` on classical hardware. Simulating $n$ qubits requires $2^n$ state vectors.
   - For an $N_{test} \times N_{train}$ similarity matrix, $N_{test} \times N_{train}$ individual quantum circuits must be executed.
2. **Qubit Allocation vs. Feature Dimensionality**:
   - The paper presents 4-qubit circuit schematics (Figures 3, 4, 5, 6, 7). Iris naturally has 4 features.
   - Wine (13), Rain (12), and Indian Liver (10) have more features than 4 qubits. The paper states that subsets of features and instances were selected for computational efficiency, but leaves the exact mapping mechanism implicit.
3. **Static Hyperparameters**:
   - The paper leaves $k$ fixed or manually chosen and does not provide an automated selection scheme.
   - Neighbor voting is uniform unweighted majority voting, which treats all $k$ nearest neighbors equally regardless of their relative quantum fidelity.
4. **Lack of Quantum Explainability**:
   - The paper evaluates classification performance but does not provide interpretability of quantum circuits, feature contributions, or Bloch-sphere dynamics.

---

## 8. Proposed Enhancements: Adaptive Hybrid Quantum KNN (AH-QKNN)

To resolve the limitations above while strictly honoring the paper's baseline, we propose the **Adaptive Hybrid Quantum KNN (AH-QKNN)**:

1. **Adaptive Feature Map Selection**:
   - Employs internal Stratified $k$-Fold Cross-Validation on `X_train` to dynamically evaluate and choose among Z, ZZ, Pauli, and Hybrid feature maps.
2. **Validation-Guided Optimal $k$ Selection**:
   - Evaluates odd neighborhood candidate values $k \in \{1, 3, 5, 7, 9\}$ on validation folds.
3. **Principled Feature-to-Qubit Encoding**:
   - Two transparent modes:
     - **Mode 1 (Compact 4-qubit)**: Top-4 features selected via Mutual Information / ANOVA F-Score, perfectly matching the paper's 4-wire circuit diagrams.
     - **Mode 2 (Full-qubit / Angle Folding)**: Native feature-to-wire mapping (up to 13 qubits) or multi-feature rotation packing ($R_x(x_1) R_y(x_2) R_z(x_3)$).
4. **Fidelity-Weighted Quantum Voting**:
   - Replaces unweighted voting with normalized quantum-fidelity weights:
     $$w_j = \frac{\mathcal{F}(\vec{x}_{test}, \vec{x}_{train}^{(j)})}{\sum_{m \in \mathcal{N}_k} \mathcal{F}(\vec{x}_{test}, \vec{x}_{train}^{(m)})}, \quad \hat{y} = \arg\max_{c} \sum_{j \in \mathcal{N}_k, y_j = c} w_j$$
5. **Comprehensive Explainability (XAI) Suite**:
   - **Classical SHAP**: TreeExplainer and KernelExplainer on classical models.
   - **Quantum XAI**:
     - Feature Sensitivity Analysis via state fidelity gradients $\nabla_{\vec{x}} \mathcal{F}(\vec{x}, \vec{x}_{train})$.
     - Interactive circuit visualizer showing gate rotations and depth.
     - Quantum neighbor contribution decomposition graph with fidelity bar charts.

---

## 9. Implementation Assumptions & Verification Log

| Item | Paper Specification | Observed Ambiguity | Documented Assumption & Solution |
| :--- | :--- | :--- | :--- |
| **Qubit Count for >4 Features** | 4-wire diagrams (Figs 3-7) | Wine (13), Rain (12), Liver (10) exceed 4 wires | Provide both 4-qubit feature-selected mode (matches paper diagrams) and full-qubit mode. |
| **Rain Dataset** | 216 records, 12 features, BOM | Standard Kaggle BOM has 145k rows | Curate the canonical 216-row 12-feature BOM benchmark subset cited in Indian academic literature. |
| **Distance Metric** | Quantum circuit in Fig. 7 | Exact formula from overlap | Overlap probability $P(\|0\dots0\rangle) = \mathcal{F}$; Distance $D_Q = 1 - \mathcal{F}$. |
| **Computational Subsets** | Mentions subsets for efficiency | Exact subset indices not given | Provide both full dataset execution and fast-execution subsets (e.g. 50-100 samples) with seed reproducibility. |
| **Simulation Backend** | PennyLane `default.qubit` | No quantum hardware claimed | Strictly use `default.qubit` statevector simulator; clearly label as quantum simulation. |

---

*Authored for the Enhanced Hybrid Quantum k-NN Classification Research Platform.*

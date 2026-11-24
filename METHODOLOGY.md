# Research Methodology: Quantum & Classical Algorithms

**Project**: Enhanced Hybrid Quantum $k$-Nearest Neighbour Classification System utilizing PennyLane  
**Reference Paper**: *Quantum k-Nearest Neighbour Classifier: Investigating Hybrid Feature map approach in Quantum Machine Learning utilizing Pennylane* (IEEE DOI: [10.1109/ICOCT64433.2025.11118830](https://doi.org/10.1109/ICOCT64433.2025.11118830))  
**Authors**: Savita Kumari Sheoran, Vikesh Yadav, Rakesh Kumar Sheoran

---

## 1. Mathematical Principles of Quantum Data Encoding

Quantum Machine Learning operates by embedding classical continuous vectors $\vec{x} = (x_1, x_2, \dots, x_d) \in \mathbb{R}^d$ into quantum state vectors $|\Phi(\vec{x})\rangle$ in a complex Hilbert space $\mathcal{H} = \mathbb{C}^{2^n}$ of dimension $2^n$, where $n$ is the number of qubits.

The transformation is performed by a parameterized unitary feature map operator $U_\Phi(\vec{x})$ initialized from the ground state $|0\dots0\rangle$:
$$|\Phi(\vec{x})\rangle = U_\Phi(\vec{x}) |0\dots0\rangle$$

---

## 2. Formulation of the Four Quantum Feature Maps

### 2.1 Z Feature Map (Pauli-Z Single-Qubit Rotations)
- **Circuit Schematic (Paper Fig. 3)**:
  For each qubit wire $i \in \{0, 1, \dots, n-1\}$:
  $$|0\rangle \xrightarrow{H} \frac{|0\rangle + |1\rangle}{\sqrt{2}} \xrightarrow{R_z(x_i)} \frac{e^{-i x_i / 2}|0\rangle + e^{i x_i / 2}|1\rangle}{\sqrt{2}}$$
- **Mathematical Form**:
  $$U_Z(\vec{x}) = \bigotimes_{i=0}^{n-1} R_z(x_i) H$$
  where $R_z(\theta) = \exp\left(-i \frac{\theta}{2} \sigma_z\right) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$.
- **Entanglement**: Zero entanglement. The output is a separable product state $|\Phi_Z(\vec{x})\rangle = \bigotimes_{i=0}^{n-1} |\phi(x_i)\rangle$.
- **Expressivity**: Captures linear feature phase offsets along the equator of the Bloch sphere.

### 2.2 ZZ Feature Map (Second-Order Pauli-Z Evolution with CNOTs)
- **Circuit Schematic (Paper Fig. 4)**:
  Combines single-qubit superposition and rotations with adjacent pairwise CNOT entangling interactions.
- **Mathematical Form**:
  $$U_{ZZ}(\vec{x}) = U_{ent}(\vec{x}) \cdot \left(\bigotimes_{i=0}^{n-1} R_z(x_i) H\right)$$
  where the second-order entangling term between adjacent wires is:
  $$U_{ent}(\vec{x}) = \prod_{i=0}^{n-2} \text{CNOT}_{(i, i+1)} \cdot R_z(x_i \cdot x_{i+1}) \cdot \text{CNOT}_{(i, i+1)} \cdot R_x(x_i)$$
- **Entanglement**: Generates non-separable multi-qubit entangled states.
- **Expressivity**: Captures quadratic interaction terms ($x_i \cdot x_{i+1}$), enabling non-linear decision boundary partitioning.

### 2.3 Pauli Feature Map (Non-Commuting Multi-Axis Rotations)
- **Circuit Schematic (Paper Fig. 5)**:
  Encodes classical inputs across non-commuting rotation generators ($X, Y, Z$) with CNOT cross-wire coupling:
- **Mathematical Form**:
  $$U_{Pauli}(\vec{x}) = \prod_{i=0}^{n-2} \text{CNOT}_{(i, i+1)} \cdot \left(\bigotimes_{i=0}^{n-1} R_z(x_i) R_y(x_i) R_x(x_i) H\right)$$
- **Expressivity**: Allows quantum states to rotate arbitrarily on the entire spherical surface of the Bloch sphere, capturing rich inter-feature angular relationships.

### 2.4 Hybrid Feature Map ("Extended Combined Feature Map")
- **Circuit Schematic (Paper Fig. 6)**:
  Cascades all three encoding mechanisms in sequence into a unified deep circuit architecture:
  $$U_{Hybrid}(\vec{x}) = U_{Pauli}(\vec{x}) \cdot U_{ZZ}(\vec{x}) \cdot U_Z(\vec{x})$$
- **Expressivity**: Maximizes Hilbert space coverage by compounding single-qubit superposition, second-order entanglement, and multi-axis rotations.

---

## 3. Quantum Distance & State Inversion Overlap (Fig. 7)

Rather than using an ancillary qubit SWAP test, the paper defines distance via the **State Inversion Overlap Circuit**:
$$\text{Circuit: } |0\dots0\rangle \xrightarrow{U(\vec{x}_{test})} |\Phi(\vec{x}_{test})\rangle \xrightarrow{U^\dagger(\vec{x}_{train})} |\Psi_{out}\rangle$$

1. **State Overlap Fidelity**:
   Measuring the projection onto the ground state $|0\dots0\rangle$ yields the transition probability:
   $$\mathcal{F}(\vec{x}_{test}, \vec{x}_{train}) = P(|0\dots0\rangle) = |\langle 0\dots0 | U^\dagger(\vec{x}_{train}) U(\vec{x}_{test}) |0\dots0\rangle|^2 = |\langle \Phi(\vec{x}_{train}) | \Phi(\vec{x}_{test}) \rangle|^2$$
   - For identical vectors ($\vec{x}_{test} = \vec{x}_{train}$), $U^\dagger(\vec{x}) U(\vec{x}) = I \implies \mathcal{F} = 1.0$.
   - For mutually orthogonal states in Hilbert space, $\mathcal{F} = 0.0$.
2. **Quantum Distance Metric**:
   $$D_Q(\vec{x}_{test}, \vec{x}_{train}) = 1 - \mathcal{F}(\vec{x}_{test}, \vec{x}_{train})$$
3. **Neighborhood Ranking**:
   The training instances are sorted in ascending order of $D_Q$ (descending order of fidelity $\mathcal{F}$), selecting the top $k$ nearest neighbors:
   $$\mathcal{N}_k(\vec{x}_{test}) = \arg\min_{j \in \{1,\dots,N_{train}\}}^{(k)} D_Q(\vec{x}_{test}, \vec{x}_{train}^{(j)})$$

---

## 4. Proposed Adaptive Hybrid Quantum KNN (AH-QKNN)

To resolve the limitation where static deep circuits induce state crowding or over-rotation on specific data geometries, AH-QKNN implements:

1. **Internal Cross-Validation on `X_train`**:
   $$\Phi^*, k^* = \arg\max_{\Phi, k} \frac{1}{V} \sum_{v=1}^V \text{F1}_{val}^{(v)}(\Phi, k)$$
   where $V=3$ stratified folds within the training split.
2. **Quantum Fidelity-Weighted Voting**:
   $$w_j = \frac{\mathcal{F}(\vec{x}_{test}, \vec{x}_{train}^{(j)})}{\sum_{m \in \mathcal{N}_k} \mathcal{F}(\vec{x}_{test}, \vec{x}_{train}^{(m)})}$$
   $$\hat{y} = \arg\max_{c \in \mathcal{C}} \sum_{j \in \mathcal{N}_k, y_j = c} w_j$$

---

## 5. Explainable AI Methodology

1. **Classical Baseline**: SHAP KernelExplainer computes additive local feature importance:
   $$f(x) = \phi_0 + \sum_{i=1}^d \phi_i$$
2. **Quantum State Overlap Gradients**:
   Feature sensitivity is derived via finite-difference numerical gradients:
   $$\nabla_{x_k} \mathcal{F}(\vec{x}, \vec{x}') \approx \frac{\mathcal{F}(\vec{x} + \epsilon \hat{e}_k, \vec{x}') - \mathcal{F}(\vec{x} - \epsilon \hat{e}_k, \vec{x}')}{2\epsilon}$$
   identifying which classical features drive quantum state separability in Hilbert space.

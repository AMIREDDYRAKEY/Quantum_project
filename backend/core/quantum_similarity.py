"""
Quantum Similarity & Inversion Overlap Engine (Phase 6)
Implements the Quantum State Inversion Overlap circuit from Figure 7 of the IEEE paper:
|0...0> ---> U(x_test) ---> U†(x_train) ---> Measure Ground State |0...0>
Transition probability P(|0...0>) yields the exact Quantum State Fidelity F.
Quantum distance is computed as D_Q = 1 - F.
"""
from typing import Optional, Tuple
import numpy as np
import pennylane as qml
from backend.core.quantum_circuits import get_feature_map


class QuantumSimilarityEngine:
    """
    Computes quantum state fidelities and distances using PennyLane.
    Provides both individual circuit execution (for verification/visualization)
    and vectorized batch statevector overlap (for fast matrix computation).
    """

    def __init__(self, feature_map_name: str = "hybrid", n_wires: int = 4, depth: int = 1):
        self.feature_map_name = feature_map_name.lower().strip()
        self.n_wires = n_wires
        self.depth = depth
        self.feature_map_func = get_feature_map(self.feature_map_name)
        self.dev = qml.device("default.qubit", wires=self.n_wires)

        # Build QNode for single pair inversion circuit (Fig. 7 in paper)
        self._build_inversion_qnode()
        # Build QNode for single sample statevector generation
        self._build_state_qnode()

    def _build_inversion_qnode(self):
        """Construct the inversion circuit QNode matching Figure 7."""
        wires = list(range(self.n_wires))
        feature_map = self.feature_map_func
        depth = self.depth

        @qml.qnode(self.dev)
        def inversion_circuit(x_test: np.ndarray, x_train: np.ndarray):
            # 1. Forward encoding of test instance: U(x_test)|0>
            feature_map(x_test, wires=wires, depth=depth)
            # 2. Adjoint encoding of training instance: U†(x_train)
            qml.adjoint(feature_map)(x_train, wires=wires, depth=depth)
            # 3. Probability measurement in computational basis
            return qml.probs(wires=wires)

        self._inversion_circuit = inversion_circuit

    def _build_state_qnode(self):
        """Construct statevector QNode for fast batch kernel computation."""
        wires = list(range(self.n_wires))
        feature_map = self.feature_map_func
        depth = self.depth

        @qml.qnode(self.dev)
        def state_circuit(x: np.ndarray):
            feature_map(x, wires=wires, depth=depth)
            return qml.state()

        self._state_circuit = state_circuit

    def compute_pair_fidelity(
        self, x1: np.ndarray, x2: np.ndarray, use_inversion_circuit: bool = True
    ) -> float:
        """
        Compute quantum state fidelity between two feature vectors.
        P(|0...0>) = |<Phi(x2)|Phi(x1)>|^2.
        F is guaranteed to be in [0.0, 1.0].
        """
        if use_inversion_circuit:
            probs = self._inversion_circuit(x1, x2)
            # Ground state |0...0> is the first element of probabilities
            fidelity = float(probs[0])
        else:
            state1 = self._state_circuit(x1)
            state2 = self._state_circuit(x2)
            inner_prod = np.vdot(state2, state1)
            fidelity = float(np.abs(inner_prod) ** 2)

        # Numerical clamping to [0, 1]
        return float(np.clip(fidelity, 0.0, 1.0))

    def compute_quantum_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute quantum distance metric: D_Q = 1 - Fidelity.
        D_Q is in [0.0, 1.0], where 0 indicates identical quantum states.
        """
        fidelity = self.compute_pair_fidelity(x1, x2)
        return float(np.clip(1.0 - fidelity, 0.0, 1.0))

    def compute_statevectors(self, X: np.ndarray) -> np.ndarray:
        """
        Compute normalized statevectors for all samples in X.
        Returns matrix of shape (N, 2^n_wires).
        """
        n_samples = len(X)
        state_dim = 2 ** self.n_wires
        states = np.zeros((n_samples, state_dim), dtype=complex)
        for i in range(n_samples):
            states[i] = self._state_circuit(X[i])
        return states

    def compute_fidelity_matrix(
        self,
        X_query: np.ndarray,
        X_train: np.ndarray,
        states_train: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized computation of the pairwise fidelity matrix between X_query and X_train.
        F_ij = |<Phi(X_train_j)|Phi(X_query_i)>|^2
        Distance matrix: D_ij = 1 - F_ij.
        """
        # Compute states
        states_query = self.compute_statevectors(X_query)
        if states_train is None:
            states_train = self.compute_statevectors(X_train)

        # Inner products: (N_query, N_train)
        # inner_products[i, j] = <states_train[j] | states_query[i]>
        inner_products = np.matmul(states_query, states_train.conj().T)
        fidelities = np.abs(inner_products) ** 2
        fidelities = np.clip(fidelities, 0.0, 1.0)
        distances = np.clip(1.0 - fidelities, 0.0, 1.0)

        return fidelities, distances

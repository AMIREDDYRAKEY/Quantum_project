"""
PennyLane Quantum Feature Maps & Quantum Distance Engine
Implements native quantum state encoding and inversion overlap state fidelity.
"""
from typing import Dict, Any, List, Optional, Tuple, Callable
import time
import numpy as np
import pennylane as qml


def angle_feature_map(features: np.ndarray, wires: List[int], depth: int = 1):
    """Simple Angle Encoding using Hadamard and RY rotations."""
    for _ in range(depth):
        for i, wire in enumerate(wires):
            if i < len(features):
                qml.Hadamard(wires=wire)
                qml.RY(float(features[i]), wires=wire)


def z_feature_map(features: np.ndarray, wires: List[int], depth: int = 1):
    """First-order Pauli-Z Feature Map (separable product state, no entanglement)."""
    for _ in range(depth):
        for i, wire in enumerate(wires):
            if i < len(features):
                qml.Hadamard(wires=wire)
                qml.RZ(float(features[i]), wires=wire)


def zz_feature_map(features: np.ndarray, wires: List[int], depth: int = 1):
    """Second-order Pauli-Z evolution with CNOT entanglement ladders."""
    n = len(wires)
    for _ in range(depth):
        for i, wire in enumerate(wires):
            if i < len(features):
                qml.Hadamard(wires=wire)
                qml.RZ(float(features[i]), wires=wire)
        for i in range(n - 1):
            if i + 1 < len(features):
                interaction = (np.pi - float(features[i])) * (np.pi - float(features[i + 1]))
                qml.CNOT(wires=[wires[i], wires[i + 1]])
                qml.RZ(interaction, wires=wires[i + 1])
                qml.CNOT(wires=[wires[i], wires[i + 1]])


def pauli_feature_map(features: np.ndarray, wires: List[int], depth: int = 1):
    """Non-commuting Pauli rotation operators (RX, RY, RZ) with entanglement."""
    n = len(wires)
    for _ in range(depth):
        for i, wire in enumerate(wires):
            if i < len(features):
                qml.Hadamard(wires=wire)
                qml.RX(float(features[i]), wires=wire)
                qml.RY(float(features[i]), wires=wire)
                qml.RZ(float(features[i]), wires=wire)
        for i in range(n - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])


def hybrid_feature_map(features: np.ndarray, wires: List[int], depth: int = 1):
    """
    IEEE ICOCT 2025 Extended Combined Feature Map.
    Cascades Z, ZZ, and Pauli layers sequentially for maximal Hilbert space expressivity.
    """
    z_feature_map(features, wires=wires, depth=depth)
    zz_feature_map(features, wires=wires, depth=depth)
    pauli_feature_map(features, wires=wires, depth=depth)


FEATURE_MAP_REGISTRY: Dict[str, Callable] = {
    "angle": angle_feature_map,
    "z": z_feature_map,
    "zz": zz_feature_map,
    "pauli": pauli_feature_map,
    "hybrid": hybrid_feature_map,
}


class QuantumSimilarityEngine:
    def __init__(self, feature_map: str = "hybrid", n_wires: int = 4, depth: int = 1):
        self.feature_map_name = feature_map.lower().strip()
        if self.feature_map_name not in FEATURE_MAP_REGISTRY:
            raise ValueError(f"Unknown feature map '{feature_map}'. Supported: {list(FEATURE_MAP_REGISTRY.keys())}")
        self.map_func = FEATURE_MAP_REGISTRY[self.feature_map_name]
        self.n_wires = n_wires
        self.depth = depth
        self.wires = list(range(n_wires))
        self.dev = qml.device("default.qubit", wires=n_wires)
        self._build_state_node()

    def _build_state_node(self):
        @qml.qnode(self.dev)
        def _circuit(x):
            self.map_func(x, wires=self.wires, depth=self.depth)
            return qml.state()
        self._state_qnode = _circuit

    def get_quantum_state(self, x: np.ndarray) -> np.ndarray:
        """Execute parameterized circuit and return full 2^N statevector."""
        return np.asarray(self._state_qnode(x), dtype=np.complex128)

    def compute_fidelity_matrix(
        self,
        X_test: np.ndarray,
        X_train: np.ndarray,
        states_train: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates State Overlap Fidelity: F(x, x') = |<psi(x)|phi(x')>|^2
        and Quantum Distance: D_Q(x, x') = 1 - F(x, x').
        Returns: (Fidelity_Matrix [N_test, N_train], Distance_Matrix [N_test, N_train])
        """
        N_train = len(X_train)
        N_test = len(X_test)

        # Precompute training state vectors
        if states_train is None:
            states_train = np.array([self.get_quantum_state(X_train[i]) for i in range(N_train)])

        # Compute test state vectors
        states_test = np.array([self.get_quantum_state(X_test[i]) for i in range(N_test)])

        # Exact quantum state overlap inner products
        inner_products = np.matmul(states_test, states_train.conj().T)
        fidelities = np.abs(inner_products) ** 2
        fidelities = np.clip(fidelities, 0.0, 1.0)
        distances = 1.0 - fidelities

        return fidelities, distances

    def get_circuit_ascii(self, sample_features: Optional[np.ndarray] = None) -> str:
        """Returns clean ASCII schematic of the quantum feature map."""
        feats = sample_features if sample_features is not None else np.array([0.5 * (i + 1) for i in range(self.n_wires)])
        @qml.qnode(self.dev)
        def dummy_circuit():
            self.map_func(feats, wires=self.wires, depth=self.depth)
            return qml.probs(wires=self.wires)
        try:
            drawer = qml.draw(dummy_circuit)
            return drawer()
        except Exception as e:
            return f"Circuit generation error: {e}"

    def get_circuit_metadata(self) -> Dict[str, Any]:
        """Inspects quantum tape to retrieve total gate and CNOT counts."""
        feats = np.zeros(self.n_wires)
        with qml.queuing.AnnotatedQueue() as q:
            self.map_func(feats, wires=self.wires, depth=self.depth)
        tape = qml.tape.QuantumScript.from_queue(q)

        gate_counts: Dict[str, int] = {}
        for op in tape.operations:
            gate_counts[op.name] = gate_counts.get(op.name, 0) + 1

        cnot_count = gate_counts.get("CNOT", 0)
        return {
            "feature_map": self.feature_map_name,
            "n_qubits": self.n_wires,
            "depth": self.depth,
            "total_gates": len(tape.operations),
            "gate_counts": gate_counts,
            "cnot_gates": cnot_count,
            "has_entanglement": cnot_count > 0,
        }

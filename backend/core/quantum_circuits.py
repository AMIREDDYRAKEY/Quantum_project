"""
Quantum Feature Maps Implementation (Phase 5)
Implements four independent quantum feature maps using PennyLane:
1. Z Feature Map (Pauli-Z rotations without entanglement, Fig. 3)
2. ZZ Feature Map (Second-order Pauli-Z evolution with CNOT entanglement, Fig. 4)
3. Pauli Feature Map (Pauli X, Y, Z rotations with entangling gates, Fig. 5)
4. Hybrid Feature Map (Extended Combined Feature Map: Z + ZZ + Pauli, Fig. 6)
"""
from typing import List, Callable, Dict, Any, Optional
import numpy as np
import pennylane as qml


def z_feature_map(features: np.ndarray, wires: List[int], depth: int = 1):
    """
    Z Feature Map (Fig. 3 in IEEE paper):
    Encodes classical features using fundamental Pauli-Z rotations without entanglement.
    For each wire i:
      |0> ---> [ H ] ---> [ RZ(x_i) ]
    Repeated for specified depth.
    """
    n_wires = len(wires)
    n_feats = len(features)
    for layer in range(depth):
        # Hadamard layer for superposition
        for w in wires:
            qml.Hadamard(wires=w)
        # Pauli-Z rotation layer
        for i, w in enumerate(wires):
            feat_idx = i % n_feats
            qml.RZ(features[feat_idx], wires=w)


def zz_feature_map(features: np.ndarray, wires: List[int], depth: int = 1):
    """
    ZZ Feature Map (Fig. 4 in IEEE paper):
    Second-order Pauli-Z evolution with CNOT entanglement.
    1. Single-qubit H and RZ encoding
    2. Pairwise entangling interactions: CNOT(i, j) -> RZ(x_i * x_j) -> CNOT(i, j)
       and RX rotation layers matching the paper's schematic.
    """
    n_wires = len(wires)
    n_feats = len(features)
    for layer in range(depth):
        # Layer 1: H and single-qubit RZ
        for w in wires:
            qml.Hadamard(wires=w)
        for i, w in enumerate(wires):
            feat_idx = i % n_feats
            qml.RZ(features[feat_idx], wires=w)

        # Layer 2: Pairwise ZZ entanglement via CNOT ladder
        if n_wires > 1:
            for i in range(n_wires - 1):
                w_ctrl = wires[i]
                w_tgt = wires[i + 1]
                idx_i = i % n_feats
                idx_j = (i + 1) % n_feats
                # Second-order interaction term
                interaction = features[idx_i] * features[idx_j]
                qml.CNOT(wires=[w_ctrl, w_tgt])
                qml.RZ(interaction, wires=w_tgt)
                qml.CNOT(wires=[w_ctrl, w_tgt])
                # Transverse field interaction
                qml.RX(features[idx_i], wires=w_ctrl)


def pauli_feature_map(features: np.ndarray, wires: List[int], depth: int = 1):
    """
    Pauli Feature Map (Fig. 5 in IEEE paper):
    Converts classical inputs by applying non-commuting Pauli rotation operators
    (RX, RY, RZ) with entangling gates across wires.
    """
    n_wires = len(wires)
    n_feats = len(features)
    for layer in range(depth):
        for w in wires:
            qml.Hadamard(wires=w)
        for i, w in enumerate(wires):
            feat_idx = i % n_feats
            qml.RX(features[feat_idx], wires=w)
            qml.RY(features[feat_idx], wires=w)
            qml.RZ(features[feat_idx], wires=w)

        # Entanglement between adjacent qubits
        if n_wires > 1:
            for i in range(n_wires - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])


def hybrid_feature_map(features: np.ndarray, wires: List[int], depth: int = 1):
    """
    Hybrid Feature Map (Extended Combined Feature Map, Fig. 6 in IEEE paper):
    Sequentially combines:
    1. Z Feature Map encoding
    2. ZZ Feature Map entangling evolution
    3. Pauli multi-axis rotations (RX, RY, RZ)
    Creating deep non-linear Hilbert space embeddings.
    """
    n_wires = len(wires)
    n_feats = len(features)
    for layer in range(depth):
        # Segment A: Z-Map
        for w in wires:
            qml.Hadamard(wires=w)
        for i, w in enumerate(wires):
            qml.RZ(features[i % n_feats], wires=w)

        # Segment B: ZZ-Map Entanglement
        if n_wires > 1:
            for i in range(n_wires - 1):
                w_ctrl = wires[i]
                w_tgt = wires[i + 1]
                idx_i = i % n_feats
                idx_j = (i + 1) % n_feats
                interaction = features[idx_i] * features[idx_j]
                qml.CNOT(wires=[w_ctrl, w_tgt])
                qml.RZ(interaction, wires=w_tgt)
                qml.CNOT(wires=[w_ctrl, w_tgt])

        # Segment C: Pauli multi-axis rotations
        for w in wires:
            qml.Hadamard(wires=w)
        for i, w in enumerate(wires):
            feat_idx = i % n_feats
            qml.RX(features[feat_idx], wires=w)
            qml.RY(features[feat_idx], wires=w)
            qml.RZ(features[feat_idx], wires=w)


# Feature Map Registry
FEATURE_MAPS: Dict[str, Callable] = {
    "z": z_feature_map,
    "zz": zz_feature_map,
    "pauli": pauli_feature_map,
    "hybrid": hybrid_feature_map,
}

FEATURE_MAP_NAMES = {
    "z": "Z Feature Map (Pauli-Z, No Entanglement)",
    "zz": "ZZ Feature Map (Second-order Evolution + CNOT)",
    "pauli": "Pauli Feature Map (RX, RY, RZ Operators)",
    "hybrid": "Hybrid Feature Map (Extended Combined: Z + ZZ + Pauli)",
}


def get_feature_map(name: str) -> Callable:
    """Retrieve feature map function by name."""
    norm_name = name.lower().strip()
    if norm_name not in FEATURE_MAPS:
        raise ValueError(
            f"Unknown feature map '{name}'. Available: {list(FEATURE_MAPS.keys())}"
        )
    return FEATURE_MAPS[norm_name]


def get_circuit_ascii(
    map_name: str,
    n_qubits: int = 4,
    features: Optional[np.ndarray] = None,
    depth: int = 1,
) -> str:
    """
    Generate clean ASCII representation of the quantum feature map circuit.
    """
    wires = list(range(n_qubits))
    feats = features if features is not None else np.array([0.5 * (i + 1) for i in range(n_qubits)])
    func = get_feature_map(map_name)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def dummy_circuit():
        func(feats, wires=wires, depth=depth)
        return qml.probs(wires=wires)

    try:
        drawer = qml.draw(dummy_circuit)
        return drawer()
    except Exception as e:
        return f"Circuit diagram generation error: {e}"


def get_circuit_metadata(map_name: str, n_qubits: int = 4, depth: int = 1) -> Dict[str, Any]:
    """
    Extract circuit properties: gate counts, entangling gate count, depth.
    """
    wires = list(range(n_qubits))
    feats = np.zeros(n_qubits)
    func = get_feature_map(map_name)

    with qml.queuing.AnnotatedQueue() as q:
        func(feats, wires=wires, depth=depth)
    tape = qml.tape.QuantumScript.from_queue(q)

    gate_names = [op.name for op in tape.operations]
    gate_counts: Dict[str, int] = {}
    for g in gate_names:
        gate_counts[g] = gate_counts.get(g, 0) + 1

    cnot_count = gate_counts.get("CNOT", 0)

    return {
        "feature_map": map_name,
        "display_name": FEATURE_MAP_NAMES.get(map_name, map_name),
        "n_qubits": n_qubits,
        "depth": depth,
        "total_gates": len(tape.operations),
        "gate_counts": gate_counts,
        "entangling_gates": cnot_count,
        "has_entanglement": cnot_count > 0,
    }

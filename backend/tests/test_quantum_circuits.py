"""Unit tests for Quantum Feature Maps (Phase 5)"""
import pytest
import numpy as np
from backend.core.quantum_circuits import (
    get_feature_map,
    get_circuit_ascii,
    get_circuit_metadata,
    FEATURE_MAPS,
)


@pytest.mark.parametrize("map_name", ["z", "zz", "pauli", "hybrid"])
def test_feature_map_retrieval(map_name):
    func = get_feature_map(map_name)
    assert callable(func)


def test_circuit_properties_and_entanglement():
    # Z feature map has NO entanglement
    z_meta = get_circuit_metadata("z", n_qubits=4, depth=1)
    assert z_meta["entangling_gates"] == 0
    assert z_meta["has_entanglement"] is False

    # ZZ feature map HAS entanglement
    zz_meta = get_circuit_metadata("zz", n_qubits=4, depth=1)
    assert zz_meta["entangling_gates"] > 0
    assert zz_meta["has_entanglement"] is True

    # Pauli feature map
    pauli_meta = get_circuit_metadata("pauli", n_qubits=4, depth=1)
    assert pauli_meta["total_gates"] > 0

    # Hybrid feature map combines all
    hybrid_meta = get_circuit_metadata("hybrid", n_qubits=4, depth=1)
    assert hybrid_meta["has_entanglement"] is True
    assert hybrid_meta["total_gates"] > z_meta["total_gates"]


def test_circuit_ascii_generation():
    ascii_circ = get_circuit_ascii("z", n_qubits=4)
    assert len(ascii_circ) > 0
    assert "0:" in ascii_circ or "H" in ascii_circ

"""Unit tests for Quantum Similarity and Overlap Inversion Circuit (Phase 6)"""
import pytest
import numpy as np
from backend.core.quantum_similarity import QuantumSimilarityEngine


def test_self_fidelity_is_unity():
    """Verify that <Phi(x)|Phi(x)>^2 = 1.0 (self fidelity)."""
    engine = QuantumSimilarityEngine(feature_map_name="hybrid", n_wires=4)
    x = np.array([0.5, -0.3, 1.2, 0.8])
    fidelity = engine.compute_pair_fidelity(x, x, use_inversion_circuit=True)
    assert np.isclose(fidelity, 1.0, atol=1e-5), f"Expected 1.0, got {fidelity}"
    dist = engine.compute_quantum_distance(x, x)
    assert np.isclose(dist, 0.0, atol=1e-5)


def test_inversion_circuit_matches_vectorized_overlap():
    """Verify single inversion circuit matches vectorized matrix calculation."""
    engine = QuantumSimilarityEngine(feature_map_name="zz", n_wires=4)
    x1 = np.array([0.1, 0.2, 0.3, 0.4])
    x2 = np.array([0.4, 0.3, 0.2, 0.1])

    # Direct inversion circuit
    fid_circ = engine.compute_pair_fidelity(x1, x2, use_inversion_circuit=True)

    # Vectorized state overlap
    fids_mat, dists_mat = engine.compute_fidelity_matrix(x1.reshape(1, -1), x2.reshape(1, -1))
    fid_mat = float(fids_mat[0, 0])

    assert np.isclose(fid_circ, fid_mat, atol=1e-5), (
        f"Circuit fidelity {fid_circ} != matrix fidelity {fid_mat}"
    )


def test_fidelity_bounded_in_unit_interval():
    engine = QuantumSimilarityEngine(feature_map_name="pauli", n_wires=4)
    np.random.seed(42)
    X = np.random.uniform(-np.pi, np.pi, size=(5, 4))
    fids, dists = engine.compute_fidelity_matrix(X, X)
    assert np.all(fids >= 0.0) and np.all(fids <= 1.0)
    assert np.all(dists >= 0.0) and np.all(dists <= 1.0)
    # Diagonal should be ~1.0
    assert np.allclose(np.diag(fids), 1.0, atol=1e-5)

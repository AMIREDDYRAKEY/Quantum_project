import os
import pytest
import pandas as pd
import numpy as np
from backend.services.dataset_service import DatasetService
from backend.preprocessing.data_preprocessor import DataPreprocessor
from backend.quantum.qknn_engine import QuantumSimilarityEngine
from backend.services.training_service import TrainingService
from backend.reports.pdf_generator import ReportGenerator
from backend.app import app

@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.normal(5.0, 1.0, 60),
        'feature2': np.random.normal(3.0, 0.5, 60),
        'feature3': np.random.normal(1.5, 0.2, 60),
        'feature4': np.random.normal(0.2, 0.1, 60),
        'target': ['class_0'] * 20 + ['class_1'] * 20 + ['class_2'] * 20
    })

def test_dataset_service_profiling(sample_df):
    meta = DatasetService.store_dataset('test_ds', sample_df, filename='test.csv')
    assert meta['dataset_id'] == 'test_ds'
    assert meta['total_rows'] == 60
    assert meta['total_columns'] == 5

    profile = DatasetService.generate_profile('test_ds')
    assert profile is not None
    assert profile['task_identification']['task_type'] == 'Classification'
    assert 'target' in profile['candidate_target_columns']
    assert len(profile['descriptive_statistics']) == 5
    assert 'feature1' in profile['correlation_matrix']

def test_data_preprocessor(sample_df):
    DatasetService.store_dataset('test_prep_ds', sample_df, filename='test_prep.csv')
    result = DataPreprocessor.configure_and_preprocess(
        dataset_id='test_prep_ds',
        target_column='target',
        test_size=0.20,
        max_features=4,
        feature_selection_method='f_classif',
        scale_method='standard',
        sample_size=60
    )
    assert result['selected_feature_count'] <= 4
    assert result['qubit_count'] <= 4
    session = DatasetService.get_session('test_prep_ds')
    prep_data = session['preprocessed_data']
    assert len(prep_data['X_train_reduced']) > 0
    assert len(prep_data['X_test_reduced']) > 0
    assert len(prep_data['y_train']) == len(prep_data['X_train_reduced'])
    assert 'X_train_angles' in prep_data
    # Check angles are in [-pi, pi]
    assert np.all(prep_data['X_train_angles'] >= -np.pi - 1e-5)
    assert np.all(prep_data['X_train_angles'] <= np.pi + 1e-5)

def test_quantum_similarity_engine():
    engine = QuantumSimilarityEngine(n_wires=2, feature_map='hybrid', depth=1)
    x1 = np.array([0.5, 1.2])
    x2 = np.array([0.5, 1.2])
    
    # Identical vectors should have fidelity close to 1.0
    fids, dists = engine.compute_fidelity_matrix(np.array([x1]), np.array([x2]))
    assert 0.99 <= fids[0, 0] <= 1.0001
    assert dists[0, 0] <= 0.01

    # Overlap matrix computation
    X_train = np.array([[0.5, 1.0], [1.5, 0.2], [2.1, 1.8]])
    X_test = np.array([[0.5, 1.0], [0.0, 0.0]])
    sim_mat, dist_mat = engine.compute_fidelity_matrix(X_test, X_train)
    assert sim_mat.shape == (2, 3)
    assert dist_mat.shape == (2, 3)
    assert np.isclose(sim_mat[0, 0], 1.0, atol=1e-3)

def test_training_service(sample_df):
    DatasetService.store_dataset('train_test_ds', sample_df, filename='train_test.csv')
    
    DataPreprocessor.configure_and_preprocess(
        dataset_id='train_test_ds',
        target_column='target',
        max_features=4,
        feature_selection_method='f_classif',
        scale_method='minmax',
        sample_size=60
    )

    train_res = TrainingService.train_and_evaluate(
        dataset_id='train_test_ds',
        k=3,
        feature_map='hybrid',
        weighted_voting=True
    )
    assert 'classical_knn' in train_res
    assert 'quantum_knn' in train_res
    assert 0.0 <= train_res['classical_knn']['accuracy'] <= 1.0
    assert 0.0 <= train_res['quantum_knn']['accuracy'] <= 1.0

    # Single prediction
    test_sample = {
        'feature1': 5.1,
        'feature2': 3.5,
        'feature3': 1.4,
        'feature4': 0.2
    }
    pred_res = TrainingService.predict_single('train_test_ds', test_sample)
    assert 'quantum_predicted_class' in pred_res
    assert 'classical_predicted_class' in pred_res
    assert len(pred_res['nearest_neighbors']) == 3

def test_pdf_report_generation(sample_df):
    DatasetService.store_dataset('pdf_test_ds', sample_df, filename='pdf_test.csv')
    DataPreprocessor.configure_and_preprocess(
        dataset_id='pdf_test_ds',
        target_column='target',
        max_features=4,
        feature_selection_method='f_classif',
        scale_method='minmax',
        sample_size=60
    )
    TrainingService.train_and_evaluate('pdf_test_ds', k=3, feature_map='angle')

    pdf_path = ReportGenerator.generate_pdf_report('pdf_test_ds')
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000

def test_flask_custom_dataset_endpoints():
    client = app.test_client()
    
    # 1. Load sample dataset
    res = client.post('/api/dataset/load-sample/iris')
    assert res.status_code == 200
    res_json = res.get_json()
    dataset_id = res_json['data']['dataset_id']
    assert dataset_id == 'sample_iris'

    # 2. Get profile
    res = client.get(f'/api/dataset/{dataset_id}/profile')
    assert res.status_code == 200
    prof = res.get_json()['data']
    assert prof['task_identification']['task_type'] == 'Classification'

    # 3. Configure
    cfg_payload = {
        'target_column': 'target',
        'n_qubits': 4,
        'scaling': 'minmax',
        'feature_selection': 'f_classif',
        'test_size': 0.25,
        'sample_size': 150
    }
    res = client.post(f'/api/dataset/{dataset_id}/configure', json=cfg_payload)
    assert res.status_code == 200

    # 4. Train
    train_payload = {
        'k': 3,
        'feature_map': 'hybrid',
        'weighted_voting': True
    }
    res = client.post(f'/api/dataset/{dataset_id}/train', json=train_payload)
    assert res.status_code == 200
    train_data = res.get_json()['data']
    assert 'classical_knn' in train_data
    assert 'quantum_knn' in train_data

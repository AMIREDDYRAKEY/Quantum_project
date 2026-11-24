import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000,
});

export const api = {
  // System Health
  getHealth: () => apiClient.get('/health'),

  // Datasets
  getDatasets: () => apiClient.get('/datasets/list'),
  getDatasetSummary: (name) => apiClient.get(`/datasets/${name}/summary`),
  getDatasetSplit: (name, params) => apiClient.post(`/datasets/${name}/split`, params),

  // Classical KNN
  trainClassicalKNN: (params) => apiClient.post('/classical/train', params),

  // Quantum KNN & Circuits
  getFeatureMaps: () => apiClient.get('/quantum/feature-maps'),
  getCircuitView: (featureMap, nQubits = 4, depth = 1) =>
    apiClient.get(`/quantum/circuit?feature_map=${featureMap}&n_qubits=${nQubits}&depth=${depth}`),
  trainQuantumKNN: (params) => apiClient.post('/quantum/train', params),

  // Adaptive Hybrid QKNN
  trainAdaptiveHybrid: (params) => apiClient.post('/adaptive/train', params),

  // Explainability
  explainClassical: (params) => apiClient.post('/xai/classical', params),
  explainQuantum: (params) => apiClient.post('/xai/quantum', params),
  getFeatureMapsTelemetry: (nQubits = 4, depth = 1) =>
    apiClient.get(`/xai/feature-maps?n_qubits=${nQubits}&depth=${depth}`),

  // Benchmarks
  getPaperResults: () => apiClient.get('/benchmarks/paper-results'),
  getLiveResults: () => apiClient.get('/benchmarks/live-results'),
  runBenchmark: (params) => apiClient.post('/benchmarks/run', params),

  // Real-time Prediction
  predictSingle: (params) => apiClient.post('/predict/single', params),
};

export default api;

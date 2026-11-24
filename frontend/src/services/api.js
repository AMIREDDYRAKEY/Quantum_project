import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL
  ? (import.meta.env.VITE_API_URL.endsWith('/api')
      ? import.meta.env.VITE_API_URL
      : `${import.meta.env.VITE_API_URL.replace(/\/$/, '')}/api`)
  : (import.meta.env.PROD
      ? 'https://quantum-project-kxpm.onrender.com/api'
      : 'http://localhost:5000/api');

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

  // Real-World Custom Dataset Ingestion, Profiling, Training, & PDF Reporting
  uploadDataset: (formData) =>
    apiClient.post('/dataset/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  loadSampleDataset: (sampleName) =>
    apiClient.post(`/dataset/load-sample/${sampleName}`),
  getDatasetProfile: (datasetId) =>
    apiClient.get(`/dataset/${datasetId}/profile`),
  getDatasetStatistics: (datasetId) =>
    apiClient.get(`/dataset/${datasetId}/statistics`),
  configureDataset: (datasetId, params) =>
    apiClient.post(`/dataset/${datasetId}/configure`, params),
  trainCustomDataset: (datasetId, params) =>
    apiClient.post(`/dataset/${datasetId}/train`, params),
  getDatasetResults: (datasetId) =>
    apiClient.get(`/dataset/${datasetId}/results`),
  predictCustomDataset: (datasetId, features) =>
    apiClient.post(`/dataset/${datasetId}/predict`, { features }),
  batchPredictCustomDataset: (datasetId, records) =>
    apiClient.post(`/dataset/${datasetId}/batch-predict`, { records }),
  getDatasetCircuit: (datasetId) =>
    apiClient.get(`/dataset/${datasetId}/quantum-circuit`),
  getReportDownloadUrl: (datasetId) =>
    `${API_BASE_URL}/dataset/${datasetId}/report`,
  downloadReportBlob: (datasetId) =>
    apiClient.get(`/dataset/${datasetId}/report`, {
      responseType: 'blob',
    }),
};

export default api;

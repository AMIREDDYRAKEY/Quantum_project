import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import ProjectOverview from './pages/ProjectOverview';
import DatasetExplorer from './pages/DatasetExplorer';
import DataPreprocessing from './pages/DataPreprocessing';
import QuantumFeatureMaps from './pages/QuantumFeatureMaps';
import QuantumCircuitVisualizer from './pages/QuantumCircuitVisualizer';
import ClassicalKNN from './pages/ClassicalKNN';
import QuantumKNN from './pages/QuantumKNN';
import AdaptiveHybridQKNN from './pages/AdaptiveHybridQKNN';
import ExplainableAI from './pages/ExplainableAI';
import ModelComparison from './pages/ModelComparison';
import PredictionPlayground from './pages/PredictionPlayground';
import ExperimentHistory from './pages/ExperimentHistory';
import CustomDatasetStudio from './pages/CustomDatasetStudio';
import { api } from './services/api';

export default function App() {
  const [activePage, setActivePage] = useState('custom-studio');
  const [activeDataset, setActiveDataset] = useState('iris');
  const [datasets, setDatasets] = useState([
    { id: 'iris', name: 'Iris Dataset', expected_records: 150 },
    { id: 'wine', name: 'Wine Dataset', expected_records: 178 },
    { id: 'rain', name: 'Rain in Australia', expected_records: 216 },
    { id: 'indian_liver', name: 'Indian Liver Patient', expected_records: 583 },
  ]);

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const res = await api.getDatasets();
      if (res.data?.datasets) {
        setDatasets(res.data.datasets);
      }
    } catch (err) {
      console.warn('Using default datasets list:', err);
    }
  };

  const renderActivePage = () => {
    switch (activePage) {
      case 'custom-studio':
        return <CustomDatasetStudio />;
      case 'dashboard':
        return <Dashboard activeDataset={activeDataset} setActivePage={setActivePage} />;
      case 'overview':
        return <ProjectOverview />;
      case 'dataset-explorer':
        return <DatasetExplorer activeDataset={activeDataset} />;
      case 'preprocessing':
        return <DataPreprocessing activeDataset={activeDataset} />;
      case 'feature-maps':
        return <QuantumFeatureMaps />;
      case 'circuit-viewer':
        return <QuantumCircuitVisualizer />;
      case 'classical-knn':
        return <ClassicalKNN activeDataset={activeDataset} />;
      case 'quantum-knn':
        return <QuantumKNN activeDataset={activeDataset} />;
      case 'adaptive-hybrid':
        return <AdaptiveHybridQKNN activeDataset={activeDataset} />;
      case 'explainable-ai':
        return <ExplainableAI activeDataset={activeDataset} />;
      case 'comparison':
        return <ModelComparison activeDataset={activeDataset} />;
      case 'prediction':
        return <PredictionPlayground activeDataset={activeDataset} />;
      case 'history':
        return <ExperimentHistory />;
      default:
        return <Dashboard activeDataset={activeDataset} setActivePage={setActivePage} />;
    }
  };

  return (
    <div className="flex h-screen w-screen flex-col bg-slate-950 text-slate-100 overflow-hidden">
      {/* Top Navbar */}
      <Navbar
        activeDataset={activeDataset}
        setActiveDataset={setActiveDataset}
        datasets={datasets}
        onRefresh={() => loadDatasets()}
      />

      {/* Main Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar activePage={activePage} setActivePage={setActivePage} />

        {/* Dynamic Page Content */}
        <main className="flex-1 overflow-y-auto p-6 bg-gradient-to-b from-slate-950 via-slate-900/40 to-slate-950">
          <div className="mx-auto max-w-7xl pb-12">
            {renderActivePage()}
          </div>
        </main>
      </div>
    </div>
  );
}

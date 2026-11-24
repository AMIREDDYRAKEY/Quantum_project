import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { PlayCircle, Cpu, CheckCircle2, RefreshCw, Sliders } from 'lucide-react';

export default function PredictionPlayground({ activeDataset }) {
  const [summary, setSummary] = useState(null);
  const [modelType, setModelType] = useState('adaptive_hybrid_qknn');
  const [featureMap, setFeatureMap] = useState('hybrid');
  const [k, setK] = useState(3);
  const [featureValues, setFeatureValues] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadDatasetInfo();
  }, [activeDataset]);

  const loadDatasetInfo = async () => {
    try {
      const res = await api.getDatasetSummary(activeDataset);
      if (res.data?.summary) {
        setSummary(res.data.summary);
        // Initialize feature sliders with median values
        const initialVals = {};
        Object.entries(res.data.summary.feature_statistics || {}).forEach(([feat, stats]) => {
          initialVals[feat] = stats.median;
        });
        setFeatureValues(initialVals);
        setPrediction(null);
      }
    } catch (err) {
      console.error('Error loading dataset info:', err);
    }
  };

  const handlePredict = async () => {
    if (!summary) return;
    setLoading(true);
    try {
      const featureArray = summary.feature_names.map((f) => featureValues[f] ?? 0);
      const res = await api.predictSingle({
        dataset: activeDataset,
        model_type: modelType,
        feature_map: featureMap,
        k: parseInt(k),
        features: featureArray,
      });
      if (res.data) {
        setPrediction(res.data);
      }
    } catch (err) {
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <PlayCircle className="h-5 w-5 text-cyan-400" />
          Real-Time Quantum Inference Playground
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Adjust classical input feature parameters using dynamic sliders and observe real-time quantum state classification.
        </p>
      </div>

      {/* Model Selection Options */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-4">
          Inference Engine Configuration
        </h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div>
            <label className="text-xs font-semibold text-slate-400 block mb-1.5">Target Classifier</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 p-2 text-xs text-white"
            >
              <option value="adaptive_hybrid_qknn">Adaptive Hybrid QKNN (Proposed)</option>
              <option value="quantum_knn">Baseline Quantum KNN</option>
              <option value="classical_knn">Classical KNN (Scikit-Learn)</option>
            </select>
          </div>

          {modelType === 'quantum_knn' && (
            <div>
              <label className="text-xs font-semibold text-slate-400 block mb-1.5">Quantum Feature Map</label>
              <select
                value={featureMap}
                onChange={(e) => setFeatureMap(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-950 p-2 text-xs text-white"
              >
                <option value="hybrid">Hybrid (Combined)</option>
                <option value="pauli">Pauli Feature Map</option>
                <option value="zz">ZZ Feature Map</option>
                <option value="z">Z Feature Map</option>
              </select>
            </div>
          )}

          <div>
            <label className="text-xs font-semibold text-slate-400 block mb-1.5">K Parameter</label>
            <select
              value={k}
              onChange={(e) => setK(parseInt(e.target.value))}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 p-2 text-xs text-white"
            >
              <option value="1">k = 1</option>
              <option value="3">k = 3</option>
              <option value="5">k = 5</option>
            </select>
          </div>
        </div>
      </div>

      {/* Interactive Feature Sliders */}
      {summary && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
              Input Feature Parameters ({summary.name})
            </h3>
            <button
              onClick={() => {
                const initialVals = {};
                Object.entries(summary.feature_statistics || {}).forEach(([feat, stats]) => {
                  initialVals[feat] = stats.median;
                });
                setFeatureValues(initialVals);
              }}
              className="text-xs font-semibold text-cyan-400 hover:text-cyan-300"
            >
              Reset to Medians
            </button>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {summary.feature_names.map((feat) => {
              const stats = summary.feature_statistics[feat] || { min: 0, max: 10, median: 5 };
              const currentVal = featureValues[feat] ?? stats.median;
              return (
                <div key={feat} className="rounded-lg bg-slate-950/70 p-3 border border-slate-800/80">
                  <div className="flex justify-between items-center text-xs mb-1.5">
                    <span className="font-semibold text-slate-300 truncate pr-2">{feat}</span>
                    <span className="font-mono text-cyan-400 font-bold">{Number(currentVal).toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={stats.min}
                    max={stats.max}
                    step={(stats.max - stats.min) / 100}
                    value={currentVal}
                    onChange={(e) =>
                      setFeatureValues({ ...featureValues, [feat]: parseFloat(e.target.value) })
                    }
                    className="w-full accent-cyan-500 cursor-pointer"
                  />
                  <div className="mt-1 flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>Min: {stats.min.toFixed(1)}</span>
                    <span>Max: {stats.max.toFixed(1)}</span>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-6 flex justify-end">
            <button
              onClick={handlePredict}
              disabled={loading}
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-500 px-6 py-3 text-xs font-bold text-white shadow-lg shadow-cyan-500/25 hover:from-cyan-400 hover:to-indigo-400 transition disabled:opacity-50"
            >
              {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <PlayCircle className="h-4 w-4" />}
              Classify Instance Real-Time
            </button>
          </div>
        </div>
      )}

      {/* Prediction Output Card */}
      {prediction && (
        <div className="rounded-xl border border-cyan-500/40 bg-gradient-to-br from-slate-900 to-cyan-950/40 p-6 backdrop-blur-md shadow-2xl space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Predicted Class Output</span>
              <h3 className="text-2xl font-black text-white mt-1 flex items-center gap-2">
                <CheckCircle2 className="h-6 w-6 text-emerald-400" />
                {prediction.predicted_label}
                <span className="text-xs font-normal text-slate-400 font-mono">
                  (Class ID: {prediction.predicted_class})
                </span>
              </h3>
            </div>
            <div className="text-right">
              <span className="text-xs text-slate-400 block">Classifier Engine</span>
              <span className="font-mono text-sm font-bold text-cyan-400">{prediction.model_type}</span>
            </div>
          </div>

          {/* Nearest Neighbors Info */}
          {prediction.neighbors && (
            <div>
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-3">
                K-Nearest Neighbors in Hilbert / Euclidean Space
              </h4>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                {prediction.neighbors.map((n) => (
                  <div key={n.rank} className="rounded-xl bg-slate-950/80 p-3.5 border border-slate-800 font-mono text-xs">
                    <div className="flex justify-between items-center text-slate-300 font-sans font-bold">
                      <span>Neighbor #{n.rank}</span>
                      <span className="text-indigo-400">Class {n.label}</span>
                    </div>
                    <div className="mt-2 space-y-1 text-[11px] text-slate-400">
                      {n.fidelity !== undefined && (
                        <div className="flex justify-between">
                          <span>Quantum Fidelity:</span>
                          <strong className="text-emerald-400">{(n.fidelity * 100).toFixed(1)}%</strong>
                        </div>
                      )}
                      {n.quantum_distance !== undefined && (
                        <div className="flex justify-between">
                          <span>Quantum Distance:</span>
                          <strong className="text-amber-400">{n.quantum_distance.toFixed(4)}</strong>
                        </div>
                      )}
                      {n.distance !== undefined && (
                        <div className="flex justify-between">
                          <span>Euclidean Distance:</span>
                          <strong className="text-cyan-400">{n.distance.toFixed(4)}</strong>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

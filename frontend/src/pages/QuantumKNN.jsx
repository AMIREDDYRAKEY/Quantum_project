import React, { useState } from 'react';
import { api } from '../services/api';
import MetricCard from '../components/MetricCard';
import ConfusionMatrix from '../components/ConfusionMatrix';
import { Atom, Play, RefreshCw, Cpu, Layers } from 'lucide-react';

export default function QuantumKNN({ activeDataset }) {
  const [featureMap, setFeatureMap] = useState('hybrid');
  const [k, setK] = useState(3);
  const [nWires, setNWires] = useState(4);
  const [depth, setDepth] = useState(1);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleTrain = async () => {
    setLoading(true);
    try {
      const res = await api.trainQuantumKNN({
        dataset: activeDataset,
        feature_map: featureMap,
        k: parseInt(k),
        n_wires: parseInt(nWires),
        depth: parseInt(depth),
      });
      if (res.data?.results) {
        setResults(res.data.results);
      }
    } catch (err) {
      console.error('Quantum KNN training error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Atom className="h-5 w-5 text-cyan-400" />
          Quantum K-Nearest Neighbour Classifier (QKNN)
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Execute actual PennyLane quantum circuits utilizing state-inversion overlap fidelities on default.qubit.
        </p>
      </div>

      {/* Control Panel */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-4">
          Quantum Circuit &amp; Classifier Parameters
        </h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-4">
          <div>
            <label className="text-xs font-semibold text-slate-400 block mb-1.5">Quantum Feature Map</label>
            <select
              value={featureMap}
              onChange={(e) => setFeatureMap(e.target.value)}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 p-2 text-xs text-white"
            >
              <option value="z">Z Feature Map (No Entanglement)</option>
              <option value="zz">ZZ Feature Map (CNOT)</option>
              <option value="pauli">Pauli Feature Map (RX, RY, RZ)</option>
              <option value="hybrid">Hybrid Feature Map (Combined)</option>
            </select>
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-400 block mb-1.5">
              K (Neighbors): <strong className="text-cyan-400">{k}</strong>
            </label>
            <input
              type="range"
              min="1"
              max="9"
              step="2"
              value={k}
              onChange={(e) => setK(parseInt(e.target.value))}
              className="w-full accent-cyan-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1">
              <span>k=1</span>
              <span>k=5</span>
              <span>k=9</span>
            </div>
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-400 block mb-1.5">Qubits (Wires)</label>
            <select
              value={nWires}
              onChange={(e) => setNWires(parseInt(e.target.value))}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 p-2 text-xs text-white"
            >
              <option value="4">4 Qubits (Paper Schematic)</option>
              <option value="6">6 Qubits</option>
            </select>
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-400 block mb-1.5">Circuit Depth</label>
            <select
              value={depth}
              onChange={(e) => setDepth(parseInt(e.target.value))}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 p-2 text-xs text-white"
            >
              <option value="1">1 Layer</option>
              <option value="2">2 Layers</option>
            </select>
          </div>
        </div>

        <div className="mt-5 flex justify-end">
          <button
            onClick={handleTrain}
            disabled={loading}
            className="flex items-center gap-2 rounded-xl bg-cyan-500 px-5 py-2.5 text-xs font-bold text-slate-950 shadow-lg shadow-cyan-500/20 hover:bg-cyan-400 transition disabled:opacity-50"
          >
            {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Execute Quantum KNN Simulation
          </button>
        </div>
      </div>

      {/* Results Section */}
      {results && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <MetricCard
              title="Accuracy"
              value={`${(results.accuracy * 100).toFixed(1)}%`}
              subtitle={`${featureMap.toUpperCase()} feature map`}
              color="cyan"
            />
            <MetricCard
              title="Weighted Precision"
              value={`${(results.precision * 100).toFixed(1)}%`}
              subtitle="Weighted precision"
              color="purple"
            />
            <MetricCard
              title="Weighted Recall"
              value={`${(results.recall * 100).toFixed(1)}%`}
              subtitle="True positive rate"
              color="emerald"
            />
            <MetricCard
              title="Weighted F1-Score"
              value={`${(results.f1_score * 100).toFixed(1)}%`}
              subtitle="Harmonic balance"
              color="amber"
            />
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
            <div className="lg:col-span-6">
              <ConfusionMatrix
                matrix={results.confusion_matrix}
                labels={results.class_labels || ['0', '1', '2']}
                title={`${results.model_type} Confusion Matrix`}
              />
            </div>

            <div className="lg:col-span-6 rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md flex flex-col justify-between">
              <div>
                <h4 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-3">
                  Quantum Execution Details
                </h4>
                <div className="space-y-3 text-xs">
                  <div className="flex justify-between border-b border-slate-800 pb-2">
                    <span className="text-slate-400">Quantum State Preparation Time:</span>
                    <span className="font-mono text-cyan-400 font-bold">{results.training_time_ms} ms</span>
                  </div>
                  <div className="flex justify-between border-b border-slate-800 pb-2">
                    <span className="text-slate-400">Quantum Overlap Inversion Time:</span>
                    <span className="font-mono text-emerald-400 font-bold">{results.inference_time_ms} ms</span>
                  </div>
                  <div className="flex justify-between border-b border-slate-800 pb-2">
                    <span className="text-slate-400">Active Wires:</span>
                    <span className="font-mono text-white">{results.n_wires} Qubits</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Distance Formulation:</span>
                    <span className="font-mono text-white">D_Q = 1 - &lang;&Phi;(x')|&Phi;(x)&rang;&sup2;</span>
                  </div>
                </div>
              </div>
              <div className="mt-4 rounded-lg bg-slate-950 p-3 text-xs text-slate-400">
                Executed via PennyLane statevector simulator default.qubit. No quantum hardware execution is claimed.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

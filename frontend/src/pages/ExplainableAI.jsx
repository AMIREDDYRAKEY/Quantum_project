import React, { useState } from 'react';
import { api } from '../services/api';
import { HelpCircle, Sparkles, Cpu, RefreshCw, BarChart2, GitCommit } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';

export default function ExplainableAI({ activeDataset }) {
  const [activeTab, setActiveTab] = useState('quantum');
  const [sampleIdx, setSampleIdx] = useState(0);
  const [featureMap, setFeatureMap] = useState('hybrid');
  const [k, setK] = useState(3);
  const [quantumXAI, setQuantumXAI] = useState(null);
  const [classicalXAI, setClassicalXAI] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleExplainQuantum = async () => {
    setLoading(true);
    try {
      const res = await api.explainQuantum({
        dataset: activeDataset,
        feature_map: featureMap,
        k: parseInt(k),
        sample_index: parseInt(sampleIdx),
        n_wires: 4,
      });
      if (res.data?.explanation) {
        setQuantumXAI(res.data.explanation);
      }
    } catch (err) {
      console.error('Quantum XAI error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleExplainClassical = async () => {
    setLoading(true);
    try {
      const res = await api.explainClassical({
        dataset: activeDataset,
        k: parseInt(k),
        sample_index: parseInt(sampleIdx),
      });
      if (res.data?.explanation) {
        setClassicalXAI(res.data.explanation);
      }
    } catch (err) {
      console.error('Classical XAI error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <HelpCircle className="h-5 w-5 text-cyan-400" />
          Explainable AI (XAI) &amp; Interpretability Suite
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          SHAP feature attributions for classical baselines &bull; Quantum state sensitivity gradients and neighbor fidelity breakdowns for QKNN.
        </p>
      </div>

      {/* Tab Switcher */}
      <div className="flex border-b border-slate-800 gap-4">
        <button
          onClick={() => setActiveTab('quantum')}
          className={`flex items-center gap-2 pb-3 text-xs font-bold transition border-b-2 ${
            activeTab === 'quantum'
              ? 'border-cyan-400 text-cyan-400'
              : 'border-transparent text-slate-400 hover:text-white'
          }`}
        >
          <Cpu className="h-4 w-4" /> Quantum Model Interpretability
        </button>
        <button
          onClick={() => setActiveTab('classical')}
          className={`flex items-center gap-2 pb-3 text-xs font-bold transition border-b-2 ${
            activeTab === 'classical'
              ? 'border-indigo-400 text-indigo-400'
              : 'border-transparent text-slate-400 hover:text-white'
          }`}
        >
          <Sparkles className="h-4 w-4" /> Classical SHAP Feature Importance
        </button>
      </div>

      {/* Quantum Tab Content */}
      {activeTab === 'quantum' && (
        <div className="space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <div className="flex flex-wrap items-center gap-4 text-xs">
              <div>
                <span className="text-slate-400 block mb-1">Feature Map:</span>
                <select
                  value={featureMap}
                  onChange={(e) => setFeatureMap(e.target.value)}
                  className="rounded-lg border border-slate-700 bg-slate-950 p-2 text-xs text-white"
                >
                  <option value="hybrid">Hybrid (Combined)</option>
                  <option value="pauli">Pauli Feature Map</option>
                  <option value="zz">ZZ Feature Map</option>
                  <option value="z">Z Feature Map</option>
                </select>
              </div>

              <div>
                <span className="text-slate-400 block mb-1">Sample Test Index:</span>
                <input
                  type="number"
                  min="0"
                  max="40"
                  value={sampleIdx}
                  onChange={(e) => setSampleIdx(e.target.value)}
                  className="w-24 rounded-lg border border-slate-700 bg-slate-950 p-1.5 text-xs text-white font-mono"
                />
              </div>

              <div>
                <span className="text-slate-400 block mb-1">K Neighbors:</span>
                <select
                  value={k}
                  onChange={(e) => setK(e.target.value)}
                  className="rounded-lg border border-slate-700 bg-slate-950 p-2 text-xs text-white"
                >
                  <option value="3">k = 3</option>
                  <option value="5">k = 5</option>
                  <option value="7">k = 7</option>
                </select>
              </div>
            </div>

            <button
              onClick={handleExplainQuantum}
              disabled={loading}
              className="flex items-center gap-2 rounded-xl bg-cyan-500 px-4 py-2 text-xs font-bold text-slate-950 hover:bg-cyan-400 transition"
            >
              {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <PlayCircle className="h-4 w-4" />}
              Explain Quantum State Overlap
            </button>
          </div>

          {quantumXAI && (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
              {/* Feature Sensitivity Bar Chart */}
              <div className="lg:col-span-7 rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-2">
                  Quantum Feature Sensitivity Gradients (&nabla;_x &fnof;)
                </h3>
                <p className="text-xs text-slate-400 mb-4">
                  Measures the rate of change of state fidelity &fnof; when each input feature is perturbed.
                </p>
                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      layout="vertical"
                      data={quantumXAI.quantum_feature_sensitivities}
                      margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                    >
                      <XAxis type="number" stroke="#64748b" tick={{ fontSize: 10 }} />
                      <YAxis dataKey="feature" type="category" stroke="#64748b" tick={{ fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }}
                        formatter={(val) => [val.toFixed(5), 'Sensitivity']}
                      />
                      <Bar dataKey="sensitivity" fill="#38bdf8" radius={[0, 4, 4, 0]}>
                        {quantumXAI.quantum_feature_sensitivities.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={index === 0 ? '#0ea5e9' : '#818cf8'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Neighbor Contributions Breakdown */}
              <div className="lg:col-span-5 rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
                      Prediction Attribution
                    </h3>
                    <p className="text-xs text-slate-400">Class Winner: <strong className="text-cyan-400">{quantumXAI.predicted_label}</strong></p>
                  </div>
                  <span className="rounded-lg bg-cyan-500/10 px-2.5 py-1 text-xs font-bold text-cyan-300 border border-cyan-500/20">
                    k = {quantumXAI.k}
                  </span>
                </div>

                <div>
                  <h4 className="text-xs font-semibold text-slate-400 mb-2">Class Vote Breakdown:</h4>
                  <div className="space-y-1.5">
                    {quantumXAI.class_votes?.map((cv) => (
                      <div key={cv.class_id} className="flex justify-between items-center bg-slate-950 p-2 rounded text-xs">
                        <span className="text-slate-300">{cv.class_label}</span>
                        <span className="font-mono font-bold text-cyan-400">{(cv.total_weight * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-semibold text-slate-400 mb-2">Nearest Quantum Neighbors:</h4>
                  <div className="space-y-2">
                    {quantumXAI.neighbor_contributions?.map((n) => (
                      <div key={n.rank} className="rounded-lg bg-slate-950 p-2.5 border border-slate-800 text-xs">
                        <div className="flex justify-between text-slate-300">
                          <span className="font-bold">Neighbor #{n.rank} (Train Idx {n.train_index})</span>
                          <span className="text-indigo-400 font-semibold">Class {n.label}</span>
                        </div>
                        <div className="mt-1 flex justify-between font-mono text-[11px] text-slate-400">
                          <span>Fidelity: <strong className="text-emerald-400">{(n.fidelity * 100).toFixed(1)}%</strong></span>
                          <span>D_Q: <strong className="text-amber-400">{n.quantum_distance.toFixed(4)}</strong></span>
                          <span>Weight: <strong className="text-cyan-400">{(n.weight * 100).toFixed(1)}%</strong></span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Classical Tab Content */}
      {activeTab === 'classical' && (
        <div className="space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <span className="text-xs text-slate-400">
              Computes additive feature attribution values for classical baseline models using SHAP KernelExplainer.
            </span>
            <button
              onClick={handleExplainClassical}
              disabled={loading}
              className="flex items-center gap-2 rounded-xl bg-indigo-600 px-4 py-2 text-xs font-bold text-white hover:bg-indigo-500 transition"
            >
              {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <PlayCircle className="h-4 w-4" />}
              Generate SHAP Explanation
            </button>
          </div>

          {classicalXAI && (
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-2">
                SHAP Feature Importance &amp; Local Attribution
              </h3>
              <p className="text-xs text-slate-400 mb-4">
                Mean absolute SHAP value impact across features on model decision boundaries.
              </p>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    layout="vertical"
                    data={classicalXAI.feature_importance}
                    margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                  >
                    <XAxis type="number" stroke="#64748b" tick={{ fontSize: 10 }} />
                    <YAxis dataKey="feature" type="category" stroke="#64748b" tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }}
                      formatter={(val) => [val.toFixed(4), 'SHAP Value']}
                    />
                    <Bar dataKey="importance" fill="#818cf8" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function PlayCircle(props) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <circle cx="12" cy="12" r="10"/>
      <polygon points="10 8 16 12 10 16 10 8"/>
    </svg>
  );
}

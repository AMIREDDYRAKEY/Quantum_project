import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { GitCompare, Download, Play, RefreshCw, Sparkles, CheckCircle2 } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';

export default function ModelComparison({ activeDataset }) {
  const [liveData, setLiveData] = useState(null);
  const [paperData, setPaperData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadAll();
  }, [activeDataset]);

  const loadAll = async () => {
    setLoading(true);
    try {
      const [liveRes, paperRes] = await Promise.all([
        api.getLiveResults(),
        api.getPaperResults(),
      ]);
      if (liveRes.data?.data) setLiveData(liveRes.data.data);
      if (paperRes.data?.data) setPaperData(paperRes.data.data);
    } catch (err) {
      console.error('Comparison load error:', err);
    } finally {
      setLoading(false);
    }
  };

  const currentLive = liveData?.[activeDataset] || {};
  const currentPaper = paperData?.table_iii_results?.[activeDataset] || {};

  const modelsList = [
    { key: 'classical_knn', label: 'Classical KNN', category: 'Classical Baseline' },
    { key: 'z_qknn', label: 'Z-QKNN', paperKey: 'Z', category: 'Paper Baseline' },
    { key: 'zz_qknn', label: 'ZZ-QKNN', paperKey: 'ZZ', category: 'Paper Baseline' },
    { key: 'pauli_qknn', label: 'Pauli-QKNN', paperKey: 'Pauli', category: 'Paper Baseline' },
    { key: 'hybrid_qknn', label: 'Hybrid-QKNN', paperKey: 'Combined', category: 'Paper Baseline' },
    { key: 'adaptive_hybrid_qknn', label: 'Adaptive Hybrid QKNN', category: 'Proposed Enhanced' },
  ];

  const chartData = modelsList.map((m) => {
    const res = currentLive[m.key] || {};
    return {
      name: m.label,
      Accuracy: (res.accuracy || 0) * 100,
      Precision: (res.precision || 0) * 100,
      Recall: (res.recall || 0) * 100,
      F1: (res.f1_score || 0) * 100,
    };
  });

  const exportCSV = () => {
    let csv = 'Dataset,Model,Accuracy,Precision,Recall,F1_Score\n';
    if (liveData) {
      Object.entries(liveData).forEach(([ds, res]) => {
        modelsList.forEach((m) => {
          const mod = res[m.key];
          if (mod) {
            csv += `${ds},${m.label},${mod.accuracy},${mod.precision},${mod.recall},${mod.f1_score}\n`;
          }
        });
      });
    }
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `quantum_knn_benchmark_comparison.csv`;
    a.click();
  };

  const exportJSON = () => {
    const blob = new Blob([JSON.stringify(liveData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `quantum_knn_benchmark_comparison.json`;
    a.click();
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <GitCompare className="h-5 w-5 text-cyan-400" />
            Empirical Model Comparison &amp; Benchmark Reports
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Side-by-side performance matrix: Classical KNN, Paper QKNN Baselines, and Proposed AH-QKNN.
          </p>
        </div>

        <div className="flex gap-2">
          <button
            onClick={exportCSV}
            className="flex items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-900 px-3.5 py-2 text-xs font-semibold text-slate-300 hover:text-white transition"
          >
            <Download className="h-4 w-4 text-cyan-400" /> Export CSV
          </button>
          <button
            onClick={exportJSON}
            className="flex items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-900 px-3.5 py-2 text-xs font-semibold text-slate-300 hover:text-white transition"
          >
            <Download className="h-4 w-4 text-indigo-400" /> Export JSON
          </button>
        </div>
      </div>

      {/* Grouped Bar Chart */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-4">
          All Evaluation Metrics &bull; {activeDataset.toUpperCase()}
        </h3>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 10 }} interval={0} angle={-15} textAnchor="end" />
              <YAxis stroke="#64748b" domain={[50, 100]} tick={{ fontSize: 10 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }}
                formatter={(val) => [`${val.toFixed(1)}%`]}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              <Bar dataKey="Accuracy" fill="#0ea5e9" radius={[3, 3, 0, 0]} />
              <Bar dataKey="Precision" fill="#818cf8" radius={[3, 3, 0, 0]} />
              <Bar dataKey="Recall" fill="#10b981" radius={[3, 3, 0, 0]} />
              <Bar dataKey="F1" fill="#f59e0b" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Comprehensive Results Table */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-4">
          Live Experimental Matrix vs. Published IEEE Paper Baseline
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-950 text-slate-400 text-[10px] uppercase">
              <tr>
                <th className="p-3">Model</th>
                <th className="p-3">Category</th>
                <th className="p-3">Live Accuracy</th>
                <th className="p-3">Live Precision</th>
                <th className="p-3">Live Recall</th>
                <th className="p-3">Live F1</th>
                <th className="p-3 text-cyan-400">Paper Table III Acc</th>
                <th className="p-3 text-cyan-400">Paper Table III F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {modelsList.map((m) => {
                const live = currentLive[m.key] || {};
                const paper = m.paperKey ? currentPaper[m.paperKey] : null;
                const isProposed = m.key === 'adaptive_hybrid_qknn';
                return (
                  <tr
                    key={m.key}
                    className={`hover:bg-slate-800/40 ${
                      isProposed ? 'bg-cyan-500/10 font-bold text-white' : 'text-slate-300'
                    }`}
                  >
                    <td className="p-3 font-sans font-bold flex items-center gap-1.5">
                      {isProposed && <Sparkles className="h-3.5 w-3.5 text-cyan-400" />}
                      {m.label}
                    </td>
                    <td className="p-3 font-sans text-slate-400 text-[11px]">{m.category}</td>
                    <td className="p-3 font-bold text-cyan-400">{live.accuracy ? (live.accuracy * 100).toFixed(1) + '%' : '--'}</td>
                    <td className="p-3">{live.precision ? (live.precision * 100).toFixed(1) + '%' : '--'}</td>
                    <td className="p-3">{live.recall ? (live.recall * 100).toFixed(1) + '%' : '--'}</td>
                    <td className="p-3 font-bold text-indigo-400">{live.f1_score ? (live.f1_score * 100).toFixed(1) + '%' : '--'}</td>
                    <td className="p-3 text-slate-400">{paper ? (paper.accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
                    <td className="p-3 text-slate-400">{paper ? (paper.f1_score * 100).toFixed(1) + '%' : 'N/A'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

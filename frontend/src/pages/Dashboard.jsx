import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import MetricCard from '../components/MetricCard';
import ConfusionMatrix from '../components/ConfusionMatrix';
import {
  Sparkles,
  Cpu,
  Layers,
  Clock,
  ArrowRight,
  TrendingUp,
  Activity,
  Play
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';

export default function Dashboard({ activeDataset, setActivePage }) {
  const [liveResults, setLiveResults] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchLiveResults();
  }, [activeDataset]);

  const fetchLiveResults = async () => {
    setLoading(true);
    try {
      const res = await api.getLiveResults();
      if (res.data?.data) {
        setLiveResults(res.data.data);
      }
    } catch (err) {
      console.error('Failed to load benchmark results:', err);
    } finally {
      setLoading(false);
    }
  };

  const currentDatasetResults = liveResults?.[activeDataset] || {};
  const ahResults = currentDatasetResults?.adaptive_hybrid_qknn || {};
  const classicalResults = currentDatasetResults?.classical_knn || {};
  const hybridResults = currentDatasetResults?.hybrid_qknn || {};
  const summary = currentDatasetResults?.summary_analysis || {};

  // Chart data comparing models on this dataset
  const chartData = [
    {
      name: 'Classical KNN',
      accuracy: (classicalResults?.accuracy || 0) * 100,
      f1: (classicalResults?.f1_score || 0) * 100,
    },
    {
      name: 'Z-QKNN',
      accuracy: (currentDatasetResults?.z_qknn?.accuracy || 0) * 100,
      f1: (currentDatasetResults?.z_qknn?.f1_score || 0) * 100,
    },
    {
      name: 'ZZ-QKNN',
      accuracy: (currentDatasetResults?.zz_qknn?.accuracy || 0) * 100,
      f1: (currentDatasetResults?.zz_qknn?.f1_score || 0) * 100,
    },
    {
      name: 'Pauli-QKNN',
      accuracy: (currentDatasetResults?.pauli_qknn?.accuracy || 0) * 100,
      f1: (currentDatasetResults?.pauli_qknn?.f1_score || 0) * 100,
    },
    {
      name: 'Hybrid-QKNN',
      accuracy: (hybridResults?.accuracy || 0) * 100,
      f1: (hybridResults?.f1_score || 0) * 100,
    },
    {
      name: 'Proposed AH-QKNN',
      accuracy: (ahResults?.accuracy || 0) * 100,
      f1: (ahResults?.f1_score || 0) * 100,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Hero Banner */}
      <div className="relative overflow-hidden rounded-2xl border border-cyan-500/30 bg-gradient-to-r from-slate-900 via-cyan-950/40 to-slate-900 p-8 shadow-2xl">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 rounded-full bg-cyan-500/20 px-3 py-1 text-xs font-bold text-cyan-300 border border-cyan-500/30 mb-3">
            <Sparkles className="h-3.5 w-3.5" />
            Research Platform &bull; Adaptive Hybrid Quantum KNN
          </div>
          <h2 className="text-2xl font-black text-white sm:text-3xl tracking-tight">
            Enhanced Quantum k-NN Classification System
          </h2>
          <p className="mt-2 text-sm text-slate-300 leading-relaxed">
            Replicating and advancing IEEE ICOCT 2025 research utilizing PennyLane quantum circuits.
            Evaluating Z, ZZ, Pauli, and Extended Hybrid Feature Maps alongside our proposed
            Validation-Guided Adaptive Hybrid Quantum KNN (AH-QKNN).
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <button
              onClick={() => setActivePage('prediction')}
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 px-5 py-2.5 text-xs font-black text-slate-950 shadow-lg shadow-emerald-500/25 hover:from-emerald-400 hover:to-cyan-400 transition cursor-pointer"
            >
              ⚡ Try Live Q-KNN Demo (Enter Values & Predict) <ArrowRight className="h-4 w-4" />
            </button>
            <button
              onClick={() => setActivePage('adaptive-hybrid')}
              className="flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-2.5 text-xs font-bold text-white hover:bg-slate-700 transition"
            >
              Explore AH-QKNN Proposed Model
            </button>
            <button
              onClick={() => setActivePage('comparison')}
              className="flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-2.5 text-xs font-bold text-white hover:bg-slate-700 transition"
            >
              Full Benchmark Matrix
            </button>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Proposed AH-QKNN Accuracy"
          value={ahResults?.accuracy ? `${(ahResults.accuracy * 100).toFixed(1)}%` : '--'}
          subtitle={`Selected: ${ahResults?.best_feature_map?.toUpperCase() || 'Hybrid'}, K=${ahResults?.best_k || 3}`}
          icon={Sparkles}
          color="cyan"
          badge="Top Performer"
        />
        <MetricCard
          title="F1-Score Gain vs Hybrid"
          value={summary?.f1_improvement !== undefined ? `${(summary.f1_improvement * 100).toFixed(1)}%` : '--'}
          subtitle={summary?.improvement_detected ? 'Measurable empirical gain' : 'Baseline matching'}
          icon={TrendingUp}
          color={summary?.improvement_detected ? 'emerald' : 'amber'}
        />
        <MetricCard
          title="Quantum Distance Metric"
          value="1 - Fidelity"
          subtitle="Derived via Inversion Overlap (Fig. 7)"
          icon={Cpu}
          color="purple"
        />
        <MetricCard
          title="Simulation Backend"
          value="default.qubit"
          subtitle="PennyLane statevector simulator"
          icon={Activity}
          color="rose"
        />
      </div>

      {/* Performance Comparison Chart & Confusion Matrix */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Performance Bar Chart */}
        <div className="lg:col-span-7 rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold uppercase tracking-wider text-white">
                Live Model Comparison &bull; {activeDataset.toUpperCase()}
              </h3>
              <p className="text-xs text-slate-400">Accuracy & F1-Score across all 6 models</p>
            </div>
            <button
              onClick={() => setActivePage('comparison')}
              className="text-xs font-semibold text-cyan-400 hover:text-cyan-300"
            >
              Full Details &rarr;
            </button>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 10 }} interval={0} angle={-15} textAnchor="end" />
                <YAxis stroke="#64748b" domain={[50, 100]} tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                  formatter={(val) => [`${val.toFixed(1)}%`]}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Bar dataKey="accuracy" name="Accuracy (%)" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
                <Bar dataKey="f1" name="F1-Score (%)" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Confusion Matrix */}
        <div className="lg:col-span-5">
          <ConfusionMatrix
            matrix={ahResults?.confusion_matrix}
            labels={ahResults?.class_labels || ['0', '1', '2']}
            title={`AH-QKNN Confusion Matrix (${activeDataset.toUpperCase()})`}
          />
        </div>
      </div>
    </div>
  );
}

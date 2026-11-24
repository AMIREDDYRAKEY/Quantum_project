import React, { useState } from 'react';
import { api } from '../services/api';
import MetricCard from '../components/MetricCard';
import ConfusionMatrix from '../components/ConfusionMatrix';
import { BarChart3, Play, Clock, CheckCircle2, RefreshCw } from 'lucide-react';

export default function ClassicalKNN({ activeDataset }) {
  const [k, setK] = useState(3);
  const [metric, setMetric] = useState('euclidean');
  const [weights, setWeights] = useState('uniform');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleTrain = async () => {
    setLoading(true);
    try {
      const res = await api.trainClassicalKNN({
        dataset: activeDataset,
        k: parseInt(k),
        metric: metric,
        weights: weights,
      });
      if (res.data?.results) {
        setResults(res.data.results);
      }
    } catch (err) {
      console.error('Classical training error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-cyan-400" />
          Classical K-Nearest Neighbour Baseline
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Scikit-learn KNN baseline model for comparative performance against quantum models.
        </p>
      </div>

      {/* Configuration Controls */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-4">
          Hyperparameter Configuration
        </h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div>
            <label className="text-xs font-semibold text-slate-400 block mb-1.5">
              K (Neighbors): <strong className="text-cyan-400">{k}</strong>
            </label>
            <input
              type="range"
              min="1"
              max="15"
              step="2"
              value={k}
              onChange={(e) => setK(parseInt(e.target.value))}
              className="w-full accent-cyan-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1">
              <span>k=1</span>
              <span>k=7</span>
              <span>k=15</span>
            </div>
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-400 block mb-1.5">Distance Metric</label>
            <select
              value={metric}
              onChange={(e) => setMetric(e.target.value)}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 p-2 text-xs text-white"
            >
              <option value="euclidean">Euclidean (L2 Norm)</option>
              <option value="manhattan">Manhattan (L1 Norm)</option>
              <option value="chebyshev">Chebyshev (L&infin; Norm)</option>
            </select>
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-400 block mb-1.5">Voting Weights</label>
            <select
              value={weights}
              onChange={(e) => setWeights(e.target.value)}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 p-2 text-xs text-white"
            >
              <option value="uniform">Uniform (Standard Majority)</option>
              <option value="distance">Inverse Distance Weighted</option>
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
            Train Classical KNN Model
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
              subtitle={`Test instances evaluated: ${results.y_true?.length}`}
              color="cyan"
            />
            <MetricCard
              title="Weighted Precision"
              value={`${(results.precision * 100).toFixed(1)}%`}
              subtitle="Macro precision weighted"
              color="purple"
            />
            <MetricCard
              title="Weighted Recall"
              value={`${(results.recall * 100).toFixed(1)}%`}
              subtitle="True positive capture rate"
              color="emerald"
            />
            <MetricCard
              title="Weighted F1-Score"
              value={`${(results.f1_score * 100).toFixed(1)}%`}
              subtitle="Harmonic mean"
              color="amber"
            />
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
            <div className="lg:col-span-6">
              <ConfusionMatrix
                matrix={results.confusion_matrix}
                labels={results.class_labels || ['0', '1', '2']}
                title="Classical KNN Confusion Matrix"
              />
            </div>

            <div className="lg:col-span-6 rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md flex flex-col justify-between">
              <div>
                <h4 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-3">
                  Execution Latency Telemetry
                </h4>
                <div className="space-y-3 text-xs">
                  <div className="flex justify-between border-b border-slate-800 pb-2">
                    <span className="text-slate-400">Training Fit Time:</span>
                    <span className="font-mono text-cyan-400 font-bold">{results.training_time_ms} ms</span>
                  </div>
                  <div className="flex justify-between border-b border-slate-800 pb-2">
                    <span className="text-slate-400">Inference Query Time:</span>
                    <span className="font-mono text-emerald-400 font-bold">{results.inference_time_ms} ms</span>
                  </div>
                  <div className="flex justify-between border-b border-slate-800 pb-2">
                    <span className="text-slate-400">Active Metric:</span>
                    <span className="capitalize font-mono text-white">{results.metric}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Voting Mechanism:</span>
                    <span className="capitalize font-mono text-white">{results.weights}</span>
                  </div>
                </div>
              </div>
              <div className="mt-4 rounded-lg bg-slate-950 p-3 text-xs text-slate-400">
                This classical model acts as the benchmark comparison for the four quantum feature maps and the proposed AH-QKNN model.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

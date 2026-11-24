import React, { useState } from 'react';
import { api } from '../services/api';
import MetricCard from '../components/MetricCard';
import ConfusionMatrix from '../components/ConfusionMatrix';
import { Sparkles, Play, RefreshCw, CheckCircle, TrendingUp, Cpu, Sliders } from 'lucide-react';

export default function AdaptiveHybridQKNN({ activeDataset }) {
  const [weighted, setWeighted] = useState(true);
  const [weightingScheme, setWeightingScheme] = useState('fidelity');
  const [candidateK, setCandidateK] = useState([1, 3, 5]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleTrain = async () => {
    setLoading(true);
    try {
      const res = await api.trainAdaptiveHybrid({
        dataset: activeDataset,
        candidate_feature_maps: ['z', 'zz', 'pauli', 'hybrid'],
        candidate_k: candidateK,
        weighted: weighted,
        weighting_scheme: weightingScheme,
        n_wires: 4,
        cv_folds: 3,
      });
      if (res.data?.results) {
        setResults(res.data.results);
      }
    } catch (err) {
      console.error('AH-QKNN training error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <div className="inline-flex items-center gap-1.5 rounded-full bg-cyan-500/10 px-3 py-1 text-xs font-semibold text-cyan-400 border border-cyan-500/20 mb-2">
          <Sparkles className="h-3.5 w-3.5" /> Proposed Model
        </div>
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          Adaptive Hybrid Quantum KNN (AH-QKNN)
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Automated Cross-Validation Feature Map Selection, Optimal K Tuning, and Fidelity-Weighted Quantum Nearest-Neighbor Voting.
        </p>
      </div>

      {/* Configuration Card */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-4">
          Optimization &amp; Weighting Controls
        </h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div>
            <label className="text-xs font-semibold text-slate-400 block mb-1.5">
              Candidate Feature Maps Tested
            </label>
            <div className="rounded-lg bg-slate-950 p-2 text-xs font-mono text-cyan-400 border border-slate-800">
              [Z, ZZ, Pauli, Hybrid]
            </div>
            <p className="text-[11px] text-slate-500 mt-1">
              Evaluated via 3-Fold CV on training split.
            </p>
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-400 block mb-1.5">
              Candidate K Neighborhoods
            </label>
            <div className="flex gap-2">
              {[1, 3, 5, 7].map((kVal) => {
                const isSelected = candidateK.includes(kVal);
                return (
                  <button
                    key={kVal}
                    onClick={() => {
                      if (isSelected && candidateK.length > 1) {
                        setCandidateK(candidateK.filter((k) => k !== kVal));
                      } else if (!isSelected) {
                        setCandidateK([...candidateK, kVal].sort());
                      }
                    }}
                    className={`rounded-lg px-3 py-1.5 text-xs font-bold transition ${
                      isSelected
                        ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20'
                        : 'bg-slate-950 text-slate-400 border border-slate-800 hover:text-white'
                    }`}
                  >
                    k={kVal}
                  </button>
                );
              })}
            </div>
            <p className="text-[11px] text-slate-500 mt-1">
              Select candidate neighborhood sizes to tune.
            </p>
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-400 block mb-1.5">
              Quantum Neighbor Voting Weight
            </label>
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={weighted}
                  onChange={(e) => setWeighted(e.target.checked)}
                  className="accent-cyan-500 rounded"
                />
                Enable Weighted Voting
              </label>
              {weighted && (
                <select
                  value={weightingScheme}
                  onChange={(e) => setWeightingScheme(e.target.value)}
                  className="rounded-lg border border-slate-700 bg-slate-950 p-1 text-xs text-white"
                >
                  <option value="fidelity">Fidelity Weighting</option>
                  <option value="inverse_distance">Inverse Distance</option>
                </select>
              )}
            </div>
            <p className="text-[11px] text-slate-500 mt-1">
              w_j &prop; Fidelity(x_test, x_train_j).
            </p>
          </div>
        </div>

        <div className="mt-5 flex justify-end">
          <button
            onClick={handleTrain}
            disabled={loading}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-500 px-5 py-2.5 text-xs font-bold text-white shadow-lg shadow-cyan-500/20 hover:from-cyan-400 hover:to-indigo-400 transition disabled:opacity-50"
          >
            {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Execute AH-QKNN Optimization &amp; Test
          </button>
        </div>
      </div>

      {/* Results Section */}
      {results && (
        <div className="space-y-6">
          {/* Optimization Decision Summary */}
          <div className="rounded-xl border border-cyan-500/30 bg-gradient-to-r from-cyan-950/30 to-indigo-950/30 p-5 backdrop-blur-md">
            <h4 className="text-sm font-bold uppercase tracking-wider text-cyan-300 mb-2 flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-cyan-400" />
              Automated Cross-Validation Selection Results
            </h4>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 mt-3">
              <div className="rounded-lg bg-slate-950/70 p-3 border border-slate-800">
                <span className="text-[11px] text-slate-400">Optimal Feature Map</span>
                <p className="text-lg font-extrabold text-cyan-400 uppercase">
                  {results.best_feature_map} Feature Map
                </p>
              </div>
              <div className="rounded-lg bg-slate-950/70 p-3 border border-slate-800">
                <span className="text-[11px] text-slate-400">Optimal K Parameter</span>
                <p className="text-lg font-extrabold text-white">K = {results.best_k}</p>
              </div>
              <div className="rounded-lg bg-slate-950/70 p-3 border border-slate-800">
                <span className="text-[11px] text-slate-400">Best Validation Score</span>
                <p className="text-lg font-extrabold text-emerald-400">
                  {results.cv_results?.best_cv_score} F1
                </p>
              </div>
              <div className="rounded-lg bg-slate-950/70 p-3 border border-slate-800">
                <span className="text-[11px] text-slate-400">Voting Weighting</span>
                <p className="text-lg font-extrabold text-indigo-400 capitalize">
                  {results.weighted ? results.weighting_scheme : 'Unweighted'}
                </p>
              </div>
            </div>
          </div>

          {/* Test Set Evaluation Metrics */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <MetricCard
              title="Held-Out Test Accuracy"
              value={`${(results.accuracy * 100).toFixed(1)}%`}
              subtitle={`Tested on ${results.y_true?.length} unseen samples`}
              color="cyan"
              badge="Empirical"
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
              subtitle="True positive recall"
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
                title="Adaptive Hybrid QKNN Confusion Matrix"
              />
            </div>

            {/* Tuning Search Log */}
            <div className="lg:col-span-6 rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
              <h4 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-3">
                Internal Cross-Validation Search Log
              </h4>
              <div className="overflow-x-auto max-h-60 overflow-y-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-slate-950 text-slate-400 text-[10px] uppercase sticky top-0">
                    <tr>
                      <th className="p-2">Map</th>
                      <th className="p-2">K</th>
                      <th className="p-2">Mean CV F1</th>
                      <th className="p-2">Std</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 text-slate-300">
                    {results.tuning_log?.map((log, idx) => (
                      <tr
                        key={idx}
                        className={`hover:bg-slate-800/40 ${
                          log.feature_map === results.best_feature_map && log.k === results.best_k
                            ? 'bg-cyan-500/10 font-bold text-cyan-300'
                            : ''
                        }`}
                      >
                        <td className="p-2 uppercase">{log.feature_map}</td>
                        <td className="p-2">k={log.k}</td>
                        <td className="p-2">{log.mean_cv_f1.toFixed(4)}</td>
                        <td className="p-2 text-slate-500">&plusmn;{log.std_cv_f1.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

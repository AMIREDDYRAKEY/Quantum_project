import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { Sliders, ShieldCheck, CheckCircle2, Split, RefreshCw } from 'lucide-react';

export default function DataPreprocessing({ activeDataset }) {
  const [splitRatio, setSplitRatio] = useState(0.20);
  const [scaleMethod, setScaleMethod] = useState('standard');
  const [nFeatures, setNFeatures] = useState(4);
  const [splitInfo, setSplitInfo] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchSplitInfo();
  }, [activeDataset, splitRatio, scaleMethod, nFeatures]);

  const fetchSplitInfo = async () => {
    setLoading(true);
    try {
      const res = await api.getDatasetSplit(activeDataset, {
        test_size: splitRatio,
        scale_method: scaleMethod,
        n_features: activeDataset !== 'iris' ? nFeatures : null,
      });
      if (res.data) {
        setSplitInfo(res.data);
      }
    } catch (err) {
      console.error('Split fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Sliders className="h-5 w-5 text-cyan-400" />
          Data Preprocessing &amp; Leakage Prevention Pipeline
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Configure train-test partitions, standard scaling, and feature-to-qubit dimensional mappings.
        </p>
      </div>

      {/* Zero Leakage Guarantee Banner */}
      <div className="rounded-xl border border-emerald-500/30 bg-emerald-950/20 p-5 backdrop-blur-md">
        <div className="flex items-start gap-3">
          <ShieldCheck className="h-6 w-6 text-emerald-400 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-bold text-white">Strict Research Zero Data Leakage Policy</h3>
            <p className="text-xs text-slate-300 mt-1 leading-relaxed">
              StandardScaler (<code className="font-mono text-emerald-300">&mu;, &sigma;</code>) is fitted <strong>strictly on the training split</strong> ({100 - splitRatio * 100}%).
              The test partition ({splitRatio * 100}%) is transformed using only the training parameters.
              Validation sets for AH-QKNN use Stratified K-Fold entirely within <code className="font-mono text-emerald-300">X_train</code>.
            </p>
          </div>
        </div>
      </div>

      {/* Interactive Controls */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        {/* Split Ratio */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-400">
            Train-Test Split Ratio
          </label>
          <div className="mt-3">
            <input
              type="range"
              min="0.10"
              max="0.40"
              step="0.05"
              value={splitRatio}
              onChange={(e) => setSplitRatio(parseFloat(e.target.value))}
              className="w-full accent-cyan-500 cursor-pointer"
            />
            <div className="mt-2 flex justify-between text-xs font-semibold">
              <span className="text-slate-400">Train: {((1 - splitRatio) * 100).toFixed(0)}%</span>
              <span className="text-cyan-400 font-bold">Test: {(splitRatio * 100).toFixed(0)}%</span>
            </div>
            <p className="mt-2 text-[11px] text-slate-500">
              IEEE baseline paper uses exact 80:20 ratio.
            </p>
          </div>
        </div>

        {/* Scaling Method */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-400">
            Feature Scaling Method
          </label>
          <div className="mt-3 space-y-2">
            {['standard', 'minmax'].map((m) => (
              <label key={m} className="flex items-center gap-2 cursor-pointer text-xs text-slate-300">
                <input
                  type="radio"
                  name="scaleMethod"
                  value={m}
                  checked={scaleMethod === m}
                  onChange={(e) => setScaleMethod(e.target.value)}
                  className="accent-cyan-500"
                />
                <span className="capitalize">{m} Scaler</span>
                {m === 'standard' && <span className="text-[10px] text-cyan-400">(Paper Baseline)</span>}
              </label>
            ))}
            <p className="text-[11px] text-slate-500 pt-1">
              Followed by bounded angle scaling <code className="font-mono text-cyan-300">2*arctan(x) &isin; [-&pi;, &pi;]</code> for quantum rotation gates.
            </p>
          </div>
        </div>

        {/* Qubit Count / Feature Selection */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-400">
            Feature Selection (Qubit Mode)
          </label>
          <div className="mt-3">
            <select
              value={nFeatures}
              onChange={(e) => setNFeatures(parseInt(e.target.value))}
              disabled={activeDataset === 'iris'}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 p-2 text-xs text-white focus:outline-none"
            >
              <option value="4">4 Features (Compact 4-Qubit Circuit Matching)</option>
              <option value="8">8 Features (High-fidelity Mode)</option>
            </select>
            <p className="mt-2 text-[11px] text-slate-500">
              {activeDataset === 'iris'
                ? 'Iris has natively 4 features (1:1 with 4 qubits).'
                : 'Selected via ANOVA F-statistic on training split.'}
            </p>
          </div>
        </div>
      </div>

      {/* Split Summary Output */}
      {splitInfo && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-3 flex items-center gap-2">
            <Split className="h-4 w-4 text-cyan-400" /> Active Partition Summary
          </h3>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div className="rounded-lg bg-slate-950 p-3 border border-slate-800">
              <span className="text-[11px] text-slate-400">Training Samples</span>
              <p className="text-xl font-extrabold text-white">{splitInfo.n_train}</p>
            </div>
            <div className="rounded-lg bg-slate-950 p-3 border border-slate-800">
              <span className="text-[11px] text-slate-400">Testing Samples</span>
              <p className="text-xl font-extrabold text-cyan-400">{splitInfo.n_test}</p>
            </div>
            <div className="rounded-lg bg-slate-950 p-3 border border-slate-800">
              <span className="text-[11px] text-slate-400">Features Selected</span>
              <p className="text-xl font-extrabold text-white">{splitInfo.n_features}</p>
            </div>
            <div className="rounded-lg bg-slate-950 p-3 border border-slate-800">
              <span className="text-[11px] text-slate-400">Target Classes</span>
              <p className="text-xl font-extrabold text-indigo-400">{splitInfo.classes?.length || 0}</p>
            </div>
          </div>

          <div className="mt-4">
            <span className="text-xs font-semibold text-slate-400">Mapped Feature Names:</span>
            <div className="mt-1 flex flex-wrap gap-2">
              {splitInfo.features?.map((f) => (
                <span key={f} className="rounded bg-slate-800 px-2.5 py-1 font-mono text-xs text-cyan-300 border border-slate-700">
                  {f}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

import React from 'react';
import { Cpu, Database, Activity, RefreshCw } from 'lucide-react';

export default function Navbar({ activeDataset = 'iris', setActiveDataset, datasets = [], onRefresh }) {
  return (
    <header className="sticky top-0 z-30 flex h-16 w-full items-center justify-between border-b border-slate-800 bg-slate-950/80 px-6 backdrop-blur-md">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-cyan-600 to-indigo-600 shadow-lg shadow-cyan-500/20">
          <Cpu className="h-6 w-6 text-white" />
        </div>
        <div>
          <h1 className="text-sm font-extrabold tracking-tight text-white flex items-center gap-2">
            Quantum k-NN Research Platform
            <span className="rounded-full bg-cyan-500/20 px-2 py-0.5 text-[10px] font-semibold text-cyan-300 border border-cyan-500/30">
              PennyLane QML
            </span>
          </h1>
          <p className="text-[11px] text-slate-400">
            Hybrid Feature Map Quantum Machine Learning Classifier
          </p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Active Dataset Selector */}
        <div className="flex items-center gap-2 rounded-xl bg-slate-900 border border-slate-800 px-3 py-1.5 shadow-sm">
          <Database className="h-4 w-4 text-cyan-400" />
          <span className="text-xs font-medium text-slate-400">Dataset:</span>
          <select
            value={activeDataset}
            onChange={(e) => setActiveDataset(e.target.value)}
            className="bg-transparent text-xs font-bold text-white focus:outline-none cursor-pointer"
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id} className="bg-slate-900 text-white">
                {d.name} ({d.expected_records} rows)
              </option>
            ))}
          </select>
        </div>

        {/* Backend Status indicator */}
        <div className="flex items-center gap-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20 px-3 py-1.5 text-xs text-emerald-400">
          <Activity className="h-4 w-4 animate-pulse" />
          <span className="font-semibold">default.qubit Active</span>
        </div>

        {onRefresh && (
          <button
            onClick={onRefresh}
            className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-800 bg-slate-900 text-slate-400 hover:text-white hover:border-slate-700 transition"
            title="Refresh System State"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        )}
      </div>
    </header>
  );
}

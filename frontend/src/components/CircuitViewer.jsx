import React from 'react';
import { Cpu, Layers, GitMerge } from 'lucide-react';

export default function CircuitViewer({ asciiDiagram, metadata, featureMapName }) {
  if (!asciiDiagram && !metadata) {
    return (
      <div className="flex h-48 items-center justify-center rounded-xl border border-slate-800 bg-slate-900/40 text-sm text-slate-500">
        Select a feature map to view circuit architecture
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-4">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Cpu className="h-5 w-5 text-cyan-400" />
            {metadata?.display_name || featureMapName || 'Quantum Circuit'}
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            PennyLane default.qubit statevector simulator architecture
          </p>
        </div>
        {metadata && (
          <div className="flex flex-wrap gap-2 text-xs">
            <span className="flex items-center gap-1 rounded-md bg-slate-800 px-2.5 py-1 text-slate-300">
              <Layers className="h-3.5 w-3.5 text-cyan-400" />
              Qubits: <strong className="text-white">{metadata.n_qubits}</strong>
            </span>
            <span className="flex items-center gap-1 rounded-md bg-slate-800 px-2.5 py-1 text-slate-300">
              Total Gates: <strong className="text-white">{metadata.total_gates}</strong>
            </span>
            <span className="flex items-center gap-1 rounded-md bg-slate-800 px-2.5 py-1 text-slate-300">
              <GitMerge className="h-3.5 w-3.5 text-indigo-400" />
              CNOT Gates: <strong className="text-white">{metadata.entangling_gates}</strong>
            </span>
            <span className={`rounded-md px-2.5 py-1 font-semibold ${metadata.has_entanglement ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30' : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'}`}>
              {metadata.has_entanglement ? 'Entangled State' : 'Separable Product State'}
            </span>
          </div>
        )}
      </div>

      {asciiDiagram && (
        <div className="overflow-x-auto rounded-lg bg-slate-950 p-4 border border-slate-800/90 shadow-inner">
          <pre className="font-mono text-xs leading-relaxed text-cyan-300 whitespace-pre">
            {asciiDiagram}
          </pre>
        </div>
      )}

      {metadata?.gate_counts && (
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold text-slate-400 mr-2">Gate Composition:</span>
          {Object.entries(metadata.gate_counts).map(([gate, count]) => (
            <span
              key={gate}
              className="inline-flex items-center gap-1 rounded bg-slate-800/80 px-2 py-0.5 text-xs text-slate-200 border border-slate-700/60 font-mono"
            >
              <strong className="text-cyan-400">{gate}</strong>: {count}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

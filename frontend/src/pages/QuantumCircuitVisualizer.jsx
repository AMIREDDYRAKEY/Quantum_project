import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { Eye, Cpu, RefreshCw, GitFork, Sliders } from 'lucide-react';
import CircuitViewer from '../components/CircuitViewer';

export default function QuantumCircuitVisualizer() {
  const [mapName, setMapName] = useState('hybrid');
  const [wires, setWires] = useState(4);
  const [depth, setDepth] = useState(1);
  const [circuit, setCircuit] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadCircuit();
  }, [mapName, wires, depth]);

  const loadCircuit = async () => {
    setLoading(true);
    try {
      const res = await api.getCircuitView(mapName, wires, depth);
      if (res.data) {
        setCircuit(res.data);
      }
    } catch (err) {
      console.error('Circuit load error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Eye className="h-5 w-5 text-cyan-400" />
          Quantum Circuit Visualizer &amp; Wire Inspector
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Inspect PennyLane quantum circuits wire by wire with gate parameters, depth, and quantum inversion schematics.
        </p>
      </div>

      {/* Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
        <div className="flex flex-wrap items-center gap-4 text-xs">
          <div>
            <span className="text-slate-400 block mb-1">Feature Map:</span>
            <select
              value={mapName}
              onChange={(e) => setMapName(e.target.value)}
              className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-white"
            >
              <option value="z">Z Feature Map (Pauli-Z)</option>
              <option value="zz">ZZ Feature Map (CNOT)</option>
              <option value="pauli">Pauli Feature Map (RX, RY, RZ)</option>
              <option value="hybrid">Hybrid Feature Map (Extended Combined)</option>
            </select>
          </div>

          <div>
            <span className="text-slate-400 block mb-1">Qubit Count:</span>
            <select
              value={wires}
              onChange={(e) => setWires(parseInt(e.target.value))}
              className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-white"
            >
              <option value="2">2 Wires</option>
              <option value="4">4 Wires (Standard)</option>
              <option value="6">6 Wires</option>
              <option value="8">8 Wires</option>
            </select>
          </div>

          <div>
            <span className="text-slate-400 block mb-1">Depth:</span>
            <select
              value={depth}
              onChange={(e) => setDepth(parseInt(e.target.value))}
              className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-white"
            >
              <option value="1">1 Layer</option>
              <option value="2">2 Layers</option>
              <option value="3">3 Layers</option>
            </select>
          </div>
        </div>

        <button
          onClick={loadCircuit}
          disabled={loading}
          className="flex items-center gap-1.5 rounded-lg bg-slate-800 px-3 py-2 text-xs font-semibold text-white hover:bg-slate-700 transition"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
          Regenerate
        </button>
      </div>

      {/* Circuit Viewer Display */}
      <CircuitViewer
        asciiDiagram={circuit?.ascii_diagram}
        metadata={circuit?.metadata}
        featureMapName={mapName}
      />

      {/* State Inversion Schematic Card */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-2 flex items-center gap-2">
          <GitFork className="h-4 w-4 text-cyan-400" />
          Quantum Overlap Inversion Circuit (IEEE Paper Fig. 7)
        </h3>
        <p className="text-xs text-slate-300 leading-relaxed mb-4">
          In QKNN, pairwise state overlap is measured directly on PennyLane using the adjoint inversion test:
        </p>
        <div className="overflow-x-auto rounded-lg bg-slate-950 p-4 border border-slate-800 font-mono text-xs text-cyan-300">
          <div>Wire 0: |0&gt; ---[ H ]---[ RZ(x_0) ]---[ RX(x_0) ]---[ RX(x'_0)&dagger; ]---[ RZ(x'_0)&dagger; ]---[ H&dagger; ]---( M )</div>
          <div className="mt-2">Wire 1: |0&gt; ---[ H ]---[ RZ(x_1) ]---[ RX(x_1) ]---[ RX(x'_1)&dagger; ]---[ RZ(x'_1)&dagger; ]---[ H&dagger; ]---( M )</div>
          <div className="mt-2">Wire 2: |0&gt; ---[ H ]---[ RZ(x_2) ]---[ RX(x_2) ]---[ RX(x'_2)&dagger; ]---[ RZ(x'_2)&dagger; ]---[ H&dagger; ]---( M )</div>
          <div className="mt-2">Wire 3: |0&gt; ---[ H ]---[ RZ(x_3) ]---[ RX(x_3) ]---[ RX(x'_3)&dagger; ]---[ RZ(x'_3)&dagger; ]---[ H&dagger; ]---( M )</div>
        </div>
        <p className="text-[11px] text-slate-400 mt-2">
          Measuring ground-state projection <code className="text-cyan-400">P(|0000&gt;)</code> gives exact fidelity &fnof;, where <code className="text-cyan-400">&fnof; = 1.0</code> for identical samples.
        </p>
      </div>
    </div>
  );
}

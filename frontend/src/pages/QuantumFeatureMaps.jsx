import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { Cpu, Layers, GitMerge, RefreshCw, Zap } from 'lucide-react';
import CircuitViewer from '../components/CircuitViewer';

export default function QuantumFeatureMaps() {
  const [selectedMap, setSelectedMap] = useState('hybrid');
  const [nQubits, setNQubits] = useState(4);
  const [depth, setDepth] = useState(1);
  const [telemetry, setTelemetry] = useState([]);
  const [circuitData, setCircuitData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchTelemetry();
  }, [nQubits, depth]);

  useEffect(() => {
    fetchCircuit();
  }, [selectedMap, nQubits, depth]);

  const fetchTelemetry = async () => {
    try {
      const res = await api.getFeatureMapsTelemetry(nQubits, depth);
      if (res.data?.feature_maps) {
        setTelemetry(res.data.feature_maps);
      }
    } catch (err) {
      console.error('Telemetry fetch error:', err);
    }
  };

  const fetchCircuit = async () => {
    setLoading(true);
    try {
      const res = await api.getCircuitView(selectedMap, nQubits, depth);
      if (res.data) {
        setCircuitData(res.data);
      }
    } catch (err) {
      console.error('Circuit view error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Cpu className="h-5 w-5 text-cyan-400" />
          Quantum Feature Maps Architecture &amp; Encoding
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Analysis of Hilbert space embeddings: Z, ZZ, Pauli, and Hybrid (Combined) Quantum Feature Maps.
        </p>
      </div>

      {/* Feature Map Selector Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { id: 'z', name: 'Z Feature Map', subtitle: 'Pauli-Z (Separable)', color: 'border-slate-700' },
          { id: 'zz', name: 'ZZ Feature Map', subtitle: 'Second-order (CNOT)', color: 'border-indigo-500/40' },
          { id: 'pauli', name: 'Pauli Feature Map', subtitle: 'Multi-axis (RX, RY, RZ)', color: 'border-purple-500/40' },
          { id: 'hybrid', name: 'Hybrid Feature Map', subtitle: 'Extended Combined', color: 'border-cyan-500/40' },
        ].map((item) => {
          const isSelected = selectedMap === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setSelectedMap(item.id)}
              className={`rounded-xl border p-4 text-left transition-all ${
                isSelected
                  ? 'border-cyan-400 bg-cyan-950/30 shadow-lg shadow-cyan-500/10 ring-1 ring-cyan-400'
                  : 'border-slate-800 bg-slate-900/60 hover:border-slate-700 hover:bg-slate-900'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-white">{item.name}</span>
                {isSelected && <Zap className="h-4 w-4 text-cyan-400" />}
              </div>
              <p className="mt-1 text-[11px] text-slate-400">{item.subtitle}</p>
            </button>
          );
        })}
      </div>

      {/* Circuit Parameters Controls */}
      <div className="flex flex-wrap items-center gap-4 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-400">Qubits (Wires):</span>
          <select
            value={nQubits}
            onChange={(e) => setNQubits(parseInt(e.target.value))}
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-white"
          >
            <option value="2">2 Qubits</option>
            <option value="4">4 Qubits (Paper Baseline)</option>
            <option value="6">6 Qubits</option>
            <option value="8">8 Qubits</option>
          </select>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-400">Circuit Depth (Repetitions):</span>
          <select
            value={depth}
            onChange={(e) => setDepth(parseInt(e.target.value))}
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-white"
          >
            <option value="1">Depth = 1 (Default)</option>
            <option value="2">Depth = 2</option>
            <option value="3">Depth = 3</option>
          </select>
        </div>
      </div>

      {/* Circuit Visualizer Component */}
      <CircuitViewer
        asciiDiagram={circuitData?.ascii_diagram}
        metadata={circuitData?.metadata}
        featureMapName={selectedMap}
      />

      {/* Comparative Telemetry Table */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-3">
          Quantum Gate Telemetry &amp; Complexity Comparison
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/60 text-slate-400 text-[10px] uppercase">
              <tr>
                <th className="p-3">Feature Map</th>
                <th className="p-3">Total Gates</th>
                <th className="p-3">CNOT Entangling Gates</th>
                <th className="p-3">Entanglement Regime</th>
                <th className="p-3">Mathematical Structure</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {telemetry.map((t) => (
                <tr key={t.feature_map} className="hover:bg-slate-800/40">
                  <td className="p-3 font-semibold text-white">{t.display_name}</td>
                  <td className="p-3 font-mono text-cyan-400 font-bold">{t.total_gates}</td>
                  <td className="p-3 font-mono text-indigo-400 font-bold">{t.entangling_gates}</td>
                  <td className="p-3">
                    <span
                      className={`inline-block rounded-full px-2.5 py-0.5 text-[10px] font-semibold ${
                        t.has_entanglement
                          ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                          : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                      }`}
                    >
                      {t.has_entanglement ? 'Entangled State' : 'Separable Product'}
                    </span>
                  </td>
                  <td className="p-3 text-slate-400 text-[11px] max-w-xs">{t.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

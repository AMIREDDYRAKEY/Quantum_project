import React from 'react';
import { BookOpen, FileText, CheckCircle2, AlertTriangle, Cpu } from 'lucide-react';

export default function ProjectOverview() {
  return (
    <div className="space-y-6 max-w-5xl">
      {/* Title Header */}
      <div>
        <div className="inline-flex items-center gap-1.5 rounded-full bg-cyan-500/10 px-3 py-1 text-xs font-semibold text-cyan-400 border border-cyan-500/20 mb-2">
          <BookOpen className="h-3.5 w-3.5" /> Research Documentation
        </div>
        <h2 className="text-2xl font-bold text-white tracking-tight">
          Quantum k-Nearest Neighbour Classifier: Investigating Hybrid Feature Map Approach
        </h2>
        <p className="mt-1 text-xs text-slate-400">
          Published in 2025 International Conference on Computing Technologies (ICOCT), Bangalore, India &bull; DOI: 10.1109/ICOCT64433.2025.11118830
        </p>
      </div>

      {/* Authors Card */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 mb-3">Authors & Affiliations</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div className="rounded-lg bg-slate-950/60 p-3 border border-slate-800">
            <h4 className="text-xs font-bold text-white">Prof. (Dr.) Savita Kumari Sheoran</h4>
            <p className="text-[11px] text-slate-400 mt-1">Professor, Dept. of Computer Science & Engineering, Indira Gandhi University Meerpur, Rewari, Haryana-INDIA</p>
          </div>
          <div className="rounded-lg bg-slate-950/60 p-3 border border-slate-800">
            <h4 className="text-xs font-bold text-white">Vikesh Yadav</h4>
            <p className="text-[11px] text-slate-400 mt-1">Research Scholar, Dept. of Computer Science & Engineering, Indira Gandhi University Meerpur, Rewari, Haryana-INDIA</p>
          </div>
          <div className="rounded-lg bg-slate-950/60 p-3 border border-slate-800">
            <h4 className="text-xs font-bold text-white">Rakesh Kumar Sheoran</h4>
            <p className="text-[11px] text-slate-400 mt-1">Assistant Professor in Physics, SIASTE, Gurugram, Haryana-INDIA</p>
          </div>
        </div>
      </div>

      {/* Abstract & Motivation */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md space-y-3">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300">Abstract Summary</h3>
        <p className="text-xs text-slate-300 leading-relaxed">
          The k-nearest neighbor algorithm (k-NN) is a simple and effective classical machine learning method that categorises unidentified test instances by identifying the k-nearest neighbours from training instances. Integrating machine learning algorithms with quantum computing offers significant benefits in complex classification regimes. This study implements Quantum k-Nearest Neighbour (QK-NN) classifiers executed on PennyLane's <code className="text-cyan-400 font-mono">default.qubit</code> statevector simulator, evaluating four feature maps: <strong>Z, ZZ, Pauli, and Combined Hybrid</strong> across four benchmark datasets (Iris, Wine, Rain, and Indian Liver Patient).
        </p>
      </div>

      {/* Four Quantum Feature Maps Formulations */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 mb-4">
          Four Quantum Feature Maps Analyzed
        </h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div className="rounded-lg bg-slate-950/80 p-4 border border-slate-800">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-bold text-cyan-400">1. Z Feature Map (Fig. 3)</h4>
              <span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded">No Entanglement</span>
            </div>
            <p className="text-xs text-slate-300">
              Encapsulates classical features using fundamental Pauli-Z rotations.
            </p>
            <div className="mt-2 rounded bg-slate-900 p-2 font-mono text-[11px] text-slate-300">
              U_Z(x) = &OElig; (R_z(x_i) &middot; H)
            </div>
          </div>

          <div className="rounded-lg bg-slate-950/80 p-4 border border-slate-800">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-bold text-indigo-400">2. ZZ Feature Map (Fig. 4)</h4>
              <span className="text-[10px] bg-indigo-500/20 text-indigo-300 px-2 py-0.5 rounded border border-indigo-500/30">CNOT Entangled</span>
            </div>
            <p className="text-xs text-slate-300">
              Second-order Pauli-Z evolution creating entanglement via CNOT ladders and pairwise interaction terms.
            </p>
            <div className="mt-2 rounded bg-slate-900 p-2 font-mono text-[11px] text-slate-300">
              U_ent = CNOT &middot; R_z(x_i &middot; x_j) &middot; CNOT
            </div>
          </div>

          <div className="rounded-lg bg-slate-950/80 p-4 border border-slate-800">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-bold text-purple-400">3. Pauli Feature Map (Fig. 5)</h4>
              <span className="text-[10px] bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded">Multi-axis Rotations</span>
            </div>
            <p className="text-xs text-slate-300">
              Applies non-commuting Pauli rotation operators (RX, RY, RZ) with inter-qubit couplings.
            </p>
            <div className="mt-2 rounded bg-slate-900 p-2 font-mono text-[11px] text-slate-300">
              U_Pauli(x) = &OElig; (R_z(x_i) &middot; R_y(x_i) &middot; R_x(x_i) &middot; H)
            </div>
          </div>

          <div className="rounded-lg bg-slate-950/80 p-4 border border-slate-800">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-bold text-emerald-400">4. Hybrid Feature Map (Fig. 6)</h4>
              <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded border border-emerald-500/30">Extended Combined</span>
            </div>
            <p className="text-xs text-slate-300">
              Sequentially cascades Z, ZZ, and Pauli encoding layers into a deep quantum circuit.
            </p>
            <div className="mt-2 rounded bg-slate-900 p-2 font-mono text-[11px] text-slate-300">
              U_Hybrid(x) = U_Pauli(x) &middot; U_ZZ(x) &middot; U_Z(x)
            </div>
          </div>
        </div>
      </div>

      {/* Quantum State Inversion Overlap */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 mb-2">
          Quantum Inversion Fidelity Distance (Fig. 7)
        </h3>
        <p className="text-xs text-slate-300 leading-relaxed">
          The quantum similarity between test sample <code className="text-cyan-400 font-mono">x_test</code> and training sample <code className="text-cyan-400 font-mono">x_train</code> is computed by executing the forward encoding followed by the adjoint encoding:
        </p>
        <div className="my-3 rounded-lg bg-slate-950 p-3 font-mono text-xs text-cyan-300 border border-slate-800">
          |0...0&gt; ---[ U(x_test) ]---[ U&dagger;(x_train) ]---[ Measure P(|0...0&gt;) ]
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          The transition probability <code className="text-cyan-400 font-mono">P(|0...0&gt;) = |&lang;&Phi;(x_train)|&Phi;(x_test)&rang;|^2</code> is the exact quantum state fidelity &fnof;. Quantum distance is then defined as <code className="text-cyan-400 font-mono">D_Q = 1 - &fnof;</code>.
        </p>
      </div>
    </div>
  );
}

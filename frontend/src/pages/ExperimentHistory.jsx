import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { History, Download, RefreshCw, FileText, CheckCircle2 } from 'lucide-react';

export default function ExperimentHistory() {
  const [historyData, setHistoryData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await api.getLiveResults();
      if (res.data?.data) {
        setHistoryData(res.data.data);
      }
    } catch (err) {
      console.error('History fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const downloadJSON = () => {
    if (!historyData) return;
    const blob = new Blob([JSON.stringify(historyData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `quantum_experiments_full_history.json`;
    a.click();
  };

  const records = [];
  if (historyData) {
    Object.entries(historyData).forEach(([ds, models]) => {
      Object.entries(models).forEach(([mKey, mVal]) => {
        if (mKey !== 'summary_analysis') {
          records.push({
            dataset: ds,
            model: mVal.model_type || mKey,
            accuracy: mVal.accuracy,
            precision: mVal.precision,
            recall: mVal.recall,
            f1_score: mVal.f1_score,
            training_time_ms: mVal.training_time_ms,
            inference_time_ms: mVal.inference_time_ms,
          });
        }
      });
    });
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <History className="h-5 w-5 text-cyan-400" />
            Experiment Run History &amp; Artifact Storage
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Audit trail of empirical trials, execution runtimes, and reproducibility logs.
          </p>
        </div>

        <button
          onClick={downloadJSON}
          className="flex items-center gap-2 rounded-xl bg-cyan-500 px-4 py-2 text-xs font-bold text-slate-950 hover:bg-cyan-400 transition"
        >
          <Download className="h-4 w-4" /> Download Complete Experiment Artifact (JSON)
        </button>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-4">
          Recorded Experiment Trials ({records.length} runs logged)
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-950 text-slate-400 text-[10px] uppercase">
              <tr>
                <th className="p-3">Dataset</th>
                <th className="p-3">Model Architecture</th>
                <th className="p-3">Accuracy</th>
                <th className="p-3">Precision</th>
                <th className="p-3">Recall</th>
                <th className="p-3">F1-Score</th>
                <th className="p-3">Train Latency</th>
                <th className="p-3">Inference Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 text-slate-300">
              {records.map((r, idx) => (
                <tr key={idx} className="hover:bg-slate-800/40">
                  <td className="p-3 uppercase font-sans font-bold text-cyan-400">{r.dataset}</td>
                  <td className="p-3 text-white font-sans">{r.model}</td>
                  <td className="p-3 text-cyan-300 font-bold">{(r.accuracy * 100).toFixed(1)}%</td>
                  <td className="p-3">{(r.precision * 100).toFixed(1)}%</td>
                  <td className="p-3">{(r.recall * 100).toFixed(1)}%</td>
                  <td className="p-3 text-indigo-400 font-bold">{(r.f1_score * 100).toFixed(1)}%</td>
                  <td className="p-3 text-slate-400">{r.training_time_ms} ms</td>
                  <td className="p-3 text-slate-400">{r.inference_time_ms} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

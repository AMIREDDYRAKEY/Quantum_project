import React from 'react';

export default function ConfusionMatrix({ matrix, labels, title = 'Confusion Matrix' }) {
  if (!matrix || matrix.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center rounded-lg border border-slate-800 bg-slate-900/50 text-sm text-slate-500">
        No confusion matrix data available
      </div>
    );
  }

  const n = matrix.length;
  const flatValues = matrix.flat();
  const maxVal = Math.max(...flatValues, 1);
  const totalSamples = flatValues.reduce((a, b) => a + b, 0);

  // Compute diagonal accuracy
  let diagonalSum = 0;
  for (let i = 0; i < n; i++) {
    diagonalSum += matrix[i][i] || 0;
  }
  const accuracyPct = ((diagonalSum / Math.max(totalSamples, 1)) * 100).toFixed(1);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h4 className="text-sm font-bold uppercase tracking-wider text-slate-200">{title}</h4>
          <p className="text-xs text-slate-400">Total samples evaluated: {totalSamples}</p>
        </div>
        <div className="rounded-lg bg-emerald-500/10 px-3 py-1 border border-emerald-500/20 text-xs font-semibold text-emerald-400">
          Accuracy: {accuracyPct}%
        </div>
      </div>

      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          <div className="text-center text-xs font-semibold uppercase tracking-wider text-cyan-400 mb-2">
            Predicted Class &rarr;
          </div>
          <table className="border-collapse mx-auto">
            <thead>
              <tr>
                <th className="p-2 text-xs font-bold text-slate-500 text-right pr-4">Actual &darr;</th>
                {labels.map((lbl, idx) => (
                  <th key={idx} className="p-2 text-center text-xs font-semibold text-slate-300 max-w-[90px] truncate" title={lbl}>
                    {lbl}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  <td className="p-2 text-right text-xs font-semibold text-slate-300 pr-4 whitespace-nowrap">
                    {labels[rowIdx] || `Class ${rowIdx}`}
                  </td>
                  {row.map((val, colIdx) => {
                    const isDiagonal = rowIdx === colIdx;
                    const intensity = Math.min(val / maxVal, 1);
                    const bgStyle = isDiagonal
                      ? `rgba(16, 185, 129, ${0.15 + intensity * 0.75})`
                      : val > 0
                      ? `rgba(244, 63, 94, ${0.15 + intensity * 0.75})`
                      : 'rgba(30, 41, 59, 0.4)';

                    return (
                      <td
                        key={colIdx}
                        style={{ backgroundColor: bgStyle }}
                        className="m-1 h-12 w-16 text-center text-sm font-bold text-white rounded border border-slate-800/80 transition-all hover:scale-105"
                      >
                        {val}
                        <span className="block text-[10px] font-normal text-slate-300 opacity-80">
                          {totalSamples > 0 ? `${((val / totalSamples) * 100).toFixed(0)}%` : '0%'}
                        </span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

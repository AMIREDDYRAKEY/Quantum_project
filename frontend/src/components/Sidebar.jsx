import React from 'react';
import {
  LayoutDashboard,
  BookOpen,
  Database,
  Sliders,
  Cpu,
  Eye,
  BarChart3,
  Atom,
  Sparkles,
  HelpCircle,
  GitCompare,
  PlayCircle,
  History,
} from 'lucide-react';

export const NAV_PAGES = [
  { id: 'dashboard', name: 'Dashboard', icon: LayoutDashboard, section: 'Core' },
  { id: 'overview', name: 'Project Overview', icon: BookOpen, section: 'Core' },
  { id: 'dataset-explorer', name: 'Dataset Explorer', icon: Database, section: 'Data' },
  { id: 'preprocessing', name: 'Data Preprocessing', icon: Sliders, section: 'Data' },
  { id: 'feature-maps', name: 'Quantum Feature Maps', icon: Cpu, section: 'Quantum' },
  { id: 'circuit-viewer', name: 'Circuit Visualizer', icon: Eye, section: 'Quantum' },
  { id: 'classical-knn', name: 'Classical KNN', icon: BarChart3, section: 'Models' },
  { id: 'quantum-knn', name: 'Quantum KNN', icon: Atom, section: 'Models' },
  { id: 'adaptive-hybrid', name: 'Adaptive Hybrid QKNN', icon: Sparkles, section: 'Models', badge: 'Proposed' },
  { id: 'explainable-ai', name: 'Explainable AI', icon: HelpCircle, section: 'Analysis' },
  { id: 'comparison', name: 'Model Comparison', icon: GitCompare, section: 'Analysis' },
  { id: 'prediction', name: 'Prediction Playground', icon: PlayCircle, section: 'Inference' },
  { id: 'history', name: 'Experiment History', icon: History, section: 'Inference' },
];

export default function Sidebar({ activePage, setActivePage }) {
  const sections = ['Core', 'Data', 'Quantum', 'Models', 'Analysis', 'Inference'];

  return (
    <aside className="w-64 flex-shrink-0 border-r border-slate-800 bg-slate-950/90 p-4 overflow-y-auto">
      <div className="space-y-6">
        {sections.map((section) => {
          const items = NAV_PAGES.filter((p) => p.section === section);
          return (
            <div key={section}>
              <div className="mb-2 px-3 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                {section}
              </div>
              <nav className="space-y-1">
                {items.map((item) => {
                  const Icon = item.icon;
                  const isActive = activePage === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => setActivePage(item.id)}
                      className={`flex w-full items-center justify-between rounded-xl px-3 py-2 text-xs font-semibold transition-all ${
                        isActive
                          ? 'bg-gradient-to-r from-cyan-500/20 to-blue-500/10 text-cyan-300 border border-cyan-500/30 shadow-sm'
                          : 'text-slate-400 hover:bg-slate-900 hover:text-slate-200'
                      }`}
                    >
                      <div className="flex items-center gap-2.5 truncate">
                        <Icon className={`h-4 w-4 flex-shrink-0 ${isActive ? 'text-cyan-400' : 'text-slate-500'}`} />
                        <span className="truncate">{item.name}</span>
                      </div>
                      {item.badge && (
                        <span className="rounded-full bg-indigo-500/20 px-1.5 py-0.2 text-[9px] font-bold text-indigo-300 border border-indigo-500/30">
                          {item.badge}
                        </span>
                      )}
                    </button>
                  );
                })}
              </nav>
            </div>
          );
        })}
      </div>
    </aside>
  );
}

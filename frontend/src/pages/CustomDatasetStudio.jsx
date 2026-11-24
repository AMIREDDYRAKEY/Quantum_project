import React, { useState } from 'react';
import { api } from '../services/api';
import {
  UploadCloud,
  FileSpreadsheet,
  BarChart2,
  Cpu,
  PlayCircle,
  FileText,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  ArrowLeft,
  RefreshCw,
  Sliders,
  Database,
  Layers,
  GitCompare,
  Download,
  Info,
  Check
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';

export default function CustomDatasetStudio() {
  const [currentStep, setCurrentStep] = useState(1);
  const [datasetId, setDatasetId] = useState(null);
  const [overview, setOverview] = useState(null);
  const [profile, setProfile] = useState(null);
  const [configInfo, setConfigInfo] = useState(null);
  const [results, setResults] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  // Configuration Form State
  const [targetColumn, setTargetColumn] = useState('');
  const [testSize, setTestSize] = useState(0.20);
  const [maxFeatures, setMaxFeatures] = useState(4);
  const [featureSelectionMethod, setFeatureSelectionMethod] = useState('f_classif');
  const [scaleMethod, setScaleMethod] = useState('standard');
  const [sampleSize, setSampleSize] = useState(200);

  // Training Form State
  const [k, setK] = useState(3);
  const [featureMap, setFeatureMap] = useState('hybrid');
  const [weightedVoting, setWeightedVoting] = useState(true);

  // Prediction State
  const [manualFeatures, setManualFeatures] = useState({});

  // ==========================================
  // STEP 1: UPLOAD & PARSE
  // ==========================================
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setLoading(true);
    setErrorMsg('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await api.uploadDataset(formData);
      if (res.data?.data) {
        setOverview(res.data.data);
        setDatasetId(res.data.data.dataset_id);
        await fetchProfile(res.data.data.dataset_id);
        setCurrentStep(2);
      }
    } catch (err) {
      console.error('Upload failed:', err);
      setErrorMsg(err.response?.data?.message || 'Failed to upload dataset. Check format (.csv, .xlsx, .xls).');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = async (sampleName) => {
    setLoading(true);
    setErrorMsg('');
    try {
      const res = await api.loadSampleDataset(sampleName);
      if (res.data?.data) {
        setOverview(res.data.data);
        setDatasetId(res.data.data.dataset_id);
        await fetchProfile(res.data.data.dataset_id);
        setCurrentStep(2);
      }
    } catch (err) {
      console.error('Load sample failed:', err);
      setErrorMsg('Failed to load sample dataset.');
    } finally {
      setLoading(false);
    }
  };

  // ==========================================
  // STEP 2: PROFILING
  // ==========================================
  const fetchProfile = async (id) => {
    try {
      const res = await api.getDatasetProfile(id);
      if (res.data?.data) {
        setProfile(res.data.data);
        if (res.data.data.recommended_target) {
          setTargetColumn(res.data.data.recommended_target);
        } else if (res.data.data.candidate_target_columns?.length > 0) {
          setTargetColumn(res.data.data.candidate_target_columns[0]);
        }
      }
    } catch (err) {
      console.error('Profile fetch failed:', err);
    }
  };

  // ==========================================
  // STEP 3: CONFIGURE & PREPROCESS
  // ==========================================
  const handleConfigure = async () => {
    if (!targetColumn) {
      setErrorMsg('Please select a target classification column.');
      return;
    }
    setLoading(true);
    setErrorMsg('');
    try {
      const res = await api.configureDataset(datasetId, {
        target_column: targetColumn,
        test_size: parseFloat(testSize),
        max_features: parseInt(maxFeatures),
        feature_selection_method: featureSelectionMethod,
        scale_method: scaleMethod,
        sample_size: parseInt(sampleSize),
        random_state: 42,
      });
      if (res.data?.data) {
        setConfigInfo(res.data.data);
        // Initialize prediction form with medians
        const initFeats = {};
        res.data.data.selected_feature_names.forEach((f) => {
          const stats = profile?.descriptive_statistics?.[f];
          initFeats[f] = stats ? stats.median : 0.0;
        });
        setManualFeatures(initFeats);
        setCurrentStep(4);
      }
    } catch (err) {
      console.error('Config failed:', err);
      setErrorMsg(err.response?.data?.message || 'Configuration failed.');
    } finally {
      setLoading(false);
    }
  };

  // ==========================================
  // STEP 4: TRAIN MODELS
  // ==========================================
  const handleTrain = async () => {
    setLoading(true);
    setErrorMsg('');
    try {
      const res = await api.trainCustomDataset(datasetId, {
        k: parseInt(k),
        feature_map: featureMap,
        weighted_voting: weightedVoting,
      });
      if (res.data?.data) {
        setResults(res.data.data);
        setCurrentStep(5);
      }
    } catch (err) {
      console.error('Training failed:', err);
      setErrorMsg(err.response?.data?.message || 'Training failed.');
    } finally {
      setLoading(false);
    }
  };

  // ==========================================
  // STEP 5: PREDICT
  // ==========================================
  const handlePredict = async () => {
    setLoading(true);
    setErrorMsg('');
    try {
      const res = await api.predictCustomDataset(datasetId, manualFeatures);
      if (res.data?.data) {
        setPrediction(res.data.data);
      }
    } catch (err) {
      console.error('Prediction failed:', err);
      setErrorMsg(err.response?.data?.message || 'Prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  const stepsList = [
    { num: 1, title: 'Upload Dataset', icon: UploadCloud },
    { num: 2, title: 'Profiling & Task', icon: BarChart2 },
    { num: 3, title: 'Quantum Config', icon: Sliders },
    { num: 4, title: 'Train & Evaluate', icon: Cpu },
    { num: 5, title: 'Results & Comparison', icon: GitCompare },
    { num: 6, title: 'Live Prediction', icon: PlayCircle },
    { num: 7, title: 'PDF Report', icon: FileText },
  ];

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="rounded-2xl border border-cyan-500/30 bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 p-6 shadow-xl">
        <div className="inline-flex items-center gap-2 rounded-full bg-cyan-500/20 px-3 py-1 text-xs font-bold text-cyan-300 border border-cyan-500/30 mb-2">
          <Database className="h-3.5 w-3.5" /> End-to-End Quantum Machine Learning Pipeline
        </div>
        <h2 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
          <UploadCloud className="h-6 w-6 text-cyan-400" />
          Real-World Dataset Ingestion & Quantum KNN Studio
        </h2>
        <p className="mt-1 text-xs text-slate-300 max-w-3xl leading-relaxed">
          Upload any classification dataset (<strong>CSV, XLSX, XLS</strong>). The platform automatically profiles statistics, identifies target suitability, safeguards qubit resources, constructs PennyLane quantum circuits, and compares Quantum KNN against Classical KNN.
        </p>
      </div>

      {/* Stepper Navigation */}
      <div className="flex overflow-x-auto rounded-2xl border border-slate-800 bg-slate-950/80 p-2 text-xs">
        {stepsList.map((s) => {
          const Icon = s.icon;
          const isDone = currentStep > s.num;
          const isCurrent = currentStep === s.num;
          return (
            <button
              key={s.num}
              onClick={() => {
                if (s.num <= 2 && overview) setCurrentStep(s.num);
                if (s.num === 3 && profile) setCurrentStep(3);
                if (s.num === 4 && configInfo) setCurrentStep(4);
                if (s.num === 5 && results) setCurrentStep(5);
                if (s.num === 6 && results) setCurrentStep(6);
                if (s.num === 7 && results) setCurrentStep(7);
              }}
              className={`flex flex-1 items-center justify-center gap-2 rounded-xl py-2.5 px-3 font-bold transition whitespace-nowrap ${
                isCurrent
                  ? 'bg-gradient-to-r from-cyan-500 to-indigo-600 text-white shadow-lg shadow-cyan-500/20'
                  : isDone
                  ? 'text-cyan-400 hover:bg-slate-900'
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              <div className={`flex h-5 w-5 items-center justify-center rounded-full text-[10px] ${isCurrent ? 'bg-white text-slate-950 font-black' : isDone ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'bg-slate-900 text-slate-500'}`}>
                {isDone ? <Check className="h-3 w-3" /> : s.num}
              </div>
              <span>{s.title}</span>
            </button>
          );
        })}
      </div>

      {errorMsg && (
        <div className="rounded-xl bg-red-500/10 border border-red-500/30 p-4 text-xs font-semibold text-red-300 flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* ========================================================= */}
      {/* STEP 1: DATASET UPLOAD & SAMPLE INGESTION                  */}
      {/* ========================================================= */}
      {currentStep === 1 && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Drag & Drop Upload Zone */}
            <div className="rounded-2xl border-2 border-dashed border-slate-700 bg-slate-900/50 p-8 text-center hover:border-cyan-400 transition">
              <UploadCloud className="h-12 w-12 text-cyan-400 mx-auto mb-3 animate-pulse" />
              <h3 className="text-base font-bold text-white">Upload Your Custom Dataset</h3>
              <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">
                Drag and drop your file here, or click to browse. Supports <strong>CSV, XLSX, XLS</strong> (up to 50 MB).
              </p>
              <label className="mt-5 inline-flex items-center gap-2 rounded-xl bg-cyan-500 px-6 py-3 text-xs font-bold text-slate-950 shadow-lg shadow-cyan-500/25 hover:bg-cyan-400 cursor-pointer transition">
                <span>Select Tabular File</span>
                <input
                  type="file"
                  accept=".csv, .xlsx, .xls"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>

            {/* 1-Click Benchmark Presets */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 flex flex-col justify-between">
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-400 font-mono">
                  Instant Test Benchmarks
                </span>
                <h3 className="text-base font-bold text-white mt-1">Or Choose a Verified Research Dataset</h3>
                <p className="text-xs text-slate-400 mt-1">
                  Test the complete pipeline immediately with one of the four reference datasets:
                </p>

                <div className="mt-4 grid grid-cols-2 gap-3">
                  {[
                    { id: 'iris', name: 'Iris Flower', rows: '150 rows', cols: '4 features' },
                    { id: 'wine', name: 'Wine Dataset', rows: '178 rows', cols: '13 features' },
                    { id: 'rain', name: 'Rain in Australia', rows: '216 rows', cols: '12 features' },
                    { id: 'indian_liver', name: 'Indian Liver Patient', rows: '583 rows', cols: '10 features' },
                  ].map((s) => (
                    <button
                      key={s.id}
                      onClick={() => handleLoadSample(s.id)}
                      disabled={loading}
                      className="rounded-xl border border-slate-800 bg-slate-950 p-3.5 text-left transition hover:border-cyan-400 hover:bg-slate-850 cursor-pointer"
                    >
                      <div className="font-bold text-xs text-white capitalize">{s.name}</div>
                      <div className="text-[11px] font-mono text-cyan-400 mt-0.5">{s.rows} &bull; {s.cols}</div>
                      <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-400 mt-2 block">
                        Load Dataset &rarr;
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              {loading && (
                <div className="mt-4 flex items-center justify-center gap-2 text-xs text-cyan-300">
                  <RefreshCw className="h-4 w-4 animate-spin" /> Ingesting and analyzing dataset...
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ========================================================= */}
      {/* STEP 2: PROFILING & TASK IDENTIFICATION                   */}
      {/* ========================================================= */}
      {currentStep === 2 && overview && profile && (
        <div className="space-y-6">
          {/* Overview KPI Cards */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
            <div className="rounded-xl bg-slate-900/80 p-3.5 border border-slate-800">
              <span className="text-[11px] text-slate-400 block">Dataset Name</span>
              <strong className="text-sm font-bold text-white truncate block">{overview.filename}</strong>
            </div>
            <div className="rounded-xl bg-slate-900/80 p-3.5 border border-slate-800">
              <span className="text-[11px] text-slate-400 block">Total Rows</span>
              <strong className="text-base font-bold text-cyan-400">{overview.total_rows.toLocaleString()}</strong>
            </div>
            <div className="rounded-xl bg-slate-900/80 p-3.5 border border-slate-800">
              <span className="text-[11px] text-slate-400 block">Total Features</span>
              <strong className="text-base font-bold text-indigo-400">{overview.total_columns}</strong>
            </div>
            <div className="rounded-xl bg-slate-900/80 p-3.5 border border-slate-800">
              <span className="text-[11px] text-slate-400 block">Missing Values</span>
              <strong className={`text-base font-bold ${overview.missing_value_count > 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
                {overview.missing_value_count}
              </strong>
            </div>
            <div className="rounded-xl bg-slate-900/80 p-3.5 border border-slate-800">
              <span className="text-[11px] text-slate-400 block">Duplicates</span>
              <strong className="text-base font-bold text-slate-300">{overview.duplicate_record_count}</strong>
            </div>
            <div className="rounded-xl bg-slate-900/80 p-3.5 border border-slate-800">
              <span className="text-[11px] text-slate-400 block">Memory Footprint</span>
              <strong className="text-base font-bold text-slate-300">{overview.memory_usage_mb} MB</strong>
            </div>
          </div>

          {/* Automated Task Identification Box */}
          <div className="rounded-2xl border border-cyan-500/40 bg-gradient-to-br from-slate-900 via-cyan-950/20 to-slate-900 p-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <span className="rounded-full bg-cyan-500/20 px-3 py-0.5 text-xs font-bold text-cyan-300 border border-cyan-500/30">
                  Automated Task Detection: {profile.task_identification?.task_type}
                </span>
                <h3 className="text-lg font-black text-white mt-1.5">
                  {profile.task_identification?.subtype || profile.task_identification?.task_type}
                </h3>
                <p className="text-xs text-slate-300 mt-1 max-w-3xl leading-relaxed">
                  {profile.task_identification?.reason}
                </p>
              </div>

              {profile.recommended_target && (
                <div className="rounded-xl bg-slate-950 p-3 border border-slate-800 text-xs">
                  <span className="text-slate-400 block text-[10px]">Detected Target Column:</span>
                  <strong className="text-emerald-400 font-mono text-sm">{profile.recommended_target}</strong>
                </div>
              )}
            </div>

            {profile.class_imbalance_warning && (
              <div className="mt-4 rounded-xl bg-amber-500/10 border border-amber-500/30 p-3 text-xs text-amber-300 flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 flex-shrink-0" />
                <span>{profile.class_imbalance_warning}</span>
              </div>
            )}

            {profile.limitations?.length > 0 && (
              <div className="mt-4 rounded-xl bg-slate-950/70 p-3.5 border border-slate-800 text-xs text-slate-400 space-y-1">
                <span className="font-bold text-slate-300 block">Scientific Disclosures & Limitations:</span>
                {profile.limitations.map((lim, idx) => (
                  <div key={idx} className="flex items-start gap-1.5 text-[11px]">
                    <span className="text-cyan-400">&bull;</span>
                    <span>{lim}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Descriptive Statistics Table */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-3">
              Feature Descriptive Statistics & Outlier Counts
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-[11px] font-bold uppercase tracking-wider text-slate-400 font-mono">
                    <th className="py-2.5 px-3">Column</th>
                    <th className="py-2.5 px-3">Type</th>
                    <th className="py-2.5 px-3">Mean</th>
                    <th className="py-2.5 px-3">Std</th>
                    <th className="py-2.5 px-3">Min</th>
                    <th className="py-2.5 px-3">Median</th>
                    <th className="py-2.5 px-3">Max</th>
                    <th className="py-2.5 px-3">Missing</th>
                    <th className="py-2.5 px-3">Outliers</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {Object.entries(profile.descriptive_statistics || {}).map(([col, s]) => (
                    <tr key={col} className="hover:bg-slate-850/50">
                      <td className="py-2.5 px-3 font-bold text-white font-sans">{col}</td>
                      <td className="py-2.5 px-3 text-cyan-400">{overview.data_types[col]}</td>
                      <td className="py-2.5 px-3 text-slate-300">{s.mean !== null ? s.mean : '--'}</td>
                      <td className="py-2.5 px-3 text-slate-300">{s.std !== null ? s.std : '--'}</td>
                      <td className="py-2.5 px-3 text-slate-300">{s.min !== null ? s.min : '--'}</td>
                      <td className="py-2.5 px-3 text-emerald-400 font-bold">{s.median !== null ? s.median : '--'}</td>
                      <td className="py-2.5 px-3 text-slate-300">{s.max !== null ? s.max : '--'}</td>
                      <td className="py-2.5 px-3 text-amber-400">{s.missing_count}</td>
                      <td className="py-2.5 px-3 text-rose-400">{s.outlier_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* First 10 Records Preview */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-3">
              Dataset Ingestion Preview (First 10 Records)
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-slate-800 text-[11px] font-bold uppercase text-slate-400">
                    {overview.column_names.map((c) => (
                      <th key={c} className="py-2 px-3">{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {overview.preview_rows.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-850/50">
                      {overview.column_names.map((c) => (
                        <td key={c} className="py-2 px-3 text-slate-300">{String(row[c] ?? '--')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="flex justify-between items-center pt-2">
            <button
              onClick={() => setCurrentStep(1)}
              className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900 px-4 py-2.5 text-xs font-bold text-slate-300 hover:text-white"
            >
              <ArrowLeft className="h-4 w-4" /> Re-upload Dataset
            </button>
            <button
              onClick={() => setCurrentStep(3)}
              className="flex items-center gap-2 rounded-xl bg-cyan-500 px-6 py-2.5 text-xs font-bold text-slate-950 shadow-lg shadow-cyan-500/25 hover:bg-cyan-400 transition"
            >
              Proceed to Quantum Configuration <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {/* ========================================================= */}
      {/* STEP 3: QUANTUM PREPROCESSING CONFIGURATION               */}
      {/* ========================================================= */}
      {currentStep === 3 && profile && (
        <div className="space-y-6">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-6 space-y-5">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-400 font-mono">
                Hardware & Simulator Preprocessing Safeguards
              </span>
              <h3 className="text-lg font-black text-white mt-1">Configure Target & Qubit Allocation</h3>
              <p className="text-xs text-slate-300 mt-1">
                Because quantum state simulation scales as $2^N$ complex amplitudes, the preprocessor uses ANOVA/PCA to automatically map your dataset into a compact, high-fidelity qubit representation.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
              {/* Target Column Selector */}
              <div className="rounded-xl bg-slate-950 p-4 border border-slate-800">
                <label className="text-xs font-bold text-slate-200 block mb-1">Target Class Column</label>
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-900 p-2 text-xs font-bold text-cyan-400 focus:outline-none"
                >
                  {profile.candidate_target_columns?.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                  {overview.column_names.filter(c => !profile.candidate_target_columns?.includes(c)).map((c) => (
                    <option key={c} value={c}>{c} (User Selected)</option>
                  ))}
                </select>
                <span className="text-[10px] text-slate-500 mt-1 block">Column to classify</span>
              </div>

              {/* Max Qubits / Features */}
              <div className="rounded-xl bg-slate-950 p-4 border border-slate-800">
                <label className="text-xs font-bold text-slate-200 block mb-1">
                  Qubit Count: <strong className="text-cyan-400">{maxFeatures} Qubits</strong>
                </label>
                <input
                  type="range"
                  min={2}
                  max={8}
                  step={1}
                  value={maxFeatures}
                  onChange={(e) => setMaxFeatures(parseInt(e.target.value))}
                  className="w-full accent-cyan-400 mt-2 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
                  <span>2 Qubits (Fast)</span>
                  <span>4 Qubits (Optimal)</span>
                  <span>8 Qubits (Deep)</span>
                </div>
              </div>

              {/* Feature Selection Method */}
              <div className="rounded-xl bg-slate-950 p-4 border border-slate-800">
                <label className="text-xs font-bold text-slate-200 block mb-1">Dimensionality Reduction</label>
                <select
                  value={featureSelectionMethod}
                  onChange={(e) => setFeatureSelectionMethod(e.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-900 p-2 text-xs font-semibold text-white focus:outline-none"
                >
                  <option value="f_classif">ANOVA F-Value (Supervised Selection)</option>
                  <option value="mutual_info">Mutual Information Criterion</option>
                  <option value="pca">Principal Component Analysis (PCA)</option>
                </select>
                <span className="text-[10px] text-slate-500 mt-1 block">Algorithm to compress features to qubits</span>
              </div>

              {/* Train / Test Split Ratio */}
              <div className="rounded-xl bg-slate-950 p-4 border border-slate-800">
                <label className="text-xs font-bold text-slate-200 block mb-1">
                  Train/Test Split: <strong className="text-indigo-400">{(100 - testSize * 100).toFixed(0)}:{ (testSize * 100).toFixed(0) }</strong>
                </label>
                <input
                  type="range"
                  min={0.10}
                  max={0.40}
                  step={0.05}
                  value={testSize}
                  onChange={(e) => setTestSize(parseFloat(e.target.value))}
                  className="w-full accent-indigo-400 mt-2 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
                  <span>90:10</span>
                  <span>80:20 (Standard)</span>
                  <span>60:40</span>
                </div>
              </div>

              {/* Feature Scaling Method */}
              <div className="rounded-xl bg-slate-950 p-4 border border-slate-800">
                <label className="text-xs font-bold text-slate-200 block mb-1">Feature Scaler</label>
                <select
                  value={scaleMethod}
                  onChange={(e) => setScaleMethod(e.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-900 p-2 text-xs font-semibold text-white focus:outline-none"
                >
                  <option value="standard">StandardScaler (Zero-mean, Unit-variance)</option>
                  <option value="minmax">MinMaxScaler ([-1, 1] range)</option>
                </select>
                <span className="text-[10px] text-slate-500 mt-1 block">Fitted strictly on training data</span>
              </div>

              {/* Sample Subsampling Guard */}
              <div className="rounded-xl bg-slate-950 p-4 border border-slate-800">
                <label className="text-xs font-bold text-slate-200 block mb-1">
                  Quantum Runtime Sample Guard: <strong className="text-cyan-400">{sampleSize} samples</strong>
                </label>
                <select
                  value={sampleSize}
                  onChange={(e) => setSampleSize(parseInt(e.target.value))}
                  className="w-full rounded-lg border border-slate-700 bg-slate-900 p-2 text-xs font-semibold text-white focus:outline-none"
                >
                  <option value="150">150 Samples (Fastest ~1s)</option>
                  <option value="200">200 Samples (Standard ~2s)</option>
                  <option value="350">350 Samples (Detailed ~5s)</option>
                  <option value="500">500 Samples (Full scale ~10s)</option>
                </select>
                <span className="text-[10px] text-slate-500 mt-1 block">Preserves stratification</span>
              </div>
            </div>

            <div className="flex justify-between items-center pt-4 border-t border-slate-800">
              <button
                onClick={() => setCurrentStep(2)}
                className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900 px-4 py-2.5 text-xs font-bold text-slate-300 hover:text-white"
              >
                <ArrowLeft className="h-4 w-4" /> Back to Profile
              </button>
              <button
                onClick={handleConfigure}
                disabled={loading}
                className="flex items-center gap-2 rounded-xl bg-cyan-500 px-6 py-2.5 text-xs font-bold text-slate-950 shadow-lg shadow-cyan-500/25 hover:bg-cyan-400 transition"
              >
                {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}
                Apply Configuration & Preprocess
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================= */}
      {/* STEP 4: MODEL TRAINING                                     */}
      {/* ========================================================= */}
      {currentStep === 4 && configInfo && (
        <div className="space-y-6">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-6 space-y-5">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-400 font-mono">
                Training Preprocessed Configuration
              </span>
              <h3 className="text-lg font-black text-white mt-1">Execute Benchmark & Quantum Simulation</h3>
              <p className="text-xs text-slate-300 mt-1">
                {configInfo.explanation}
              </p>
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              {/* K Parameter */}
              <div className="rounded-xl bg-slate-950 p-4 border border-slate-800">
                <label className="text-xs font-bold text-slate-200 block mb-1">K Value (Nearest Neighbors)</label>
                <select
                  value={k}
                  onChange={(e) => setK(parseInt(e.target.value))}
                  className="w-full rounded-lg border border-slate-700 bg-slate-900 p-2 text-xs font-bold text-cyan-400 focus:outline-none"
                >
                  <option value="1">k = 1</option>
                  <option value="3">k = 3 (Standard)</option>
                  <option value="5">k = 5</option>
                  <option value="7">k = 7</option>
                </select>
                <span className="text-[10px] text-slate-500 mt-1 block">Neighborhood cardinality in Hilbert space</span>
              </div>

              {/* Feature Map */}
              <div className="rounded-xl bg-slate-950 p-4 border border-slate-800">
                <label className="text-xs font-bold text-slate-200 block mb-1">Quantum Feature Map</label>
                <select
                  value={featureMap}
                  onChange={(e) => setFeatureMap(e.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-900 p-2 text-xs font-bold text-white focus:outline-none"
                >
                  <option value="hybrid">Hybrid Feature Map (Z + ZZ + Pauli)</option>
                  <option value="pauli">Pauli Feature Map (RX, RY, RZ)</option>
                  <option value="zz">ZZ Feature Map (CNOT Entanglement)</option>
                  <option value="z">Z Feature Map (Product State)</option>
                  <option value="angle">Simple Angle Encoding</option>
                </select>
                <span className="text-[10px] text-slate-500 mt-1 block">Circuit ansatz for state preparation</span>
              </div>

              {/* Weighted Voting Toggle */}
              <div className="rounded-xl bg-slate-950 p-4 border border-slate-800">
                <label className="text-xs font-bold text-slate-200 block mb-1">Voting Weighting</label>
                <div className="flex items-center gap-3 mt-2">
                  <input
                    type="checkbox"
                    id="weighted-voting"
                    checked={weightedVoting}
                    onChange={(e) => setWeightedVoting(e.target.checked)}
                    className="h-4 w-4 accent-cyan-400 cursor-pointer"
                  />
                  <label htmlFor="weighted-voting" className="text-xs font-semibold text-slate-300 cursor-pointer">
                    Fidelity-Weighted Voting (w ∝ F)
                  </label>
                </div>
                <span className="text-[10px] text-slate-500 mt-1.5 block">Weights votes by exact quantum state overlap</span>
              </div>
            </div>

            <div className="flex justify-between items-center pt-4 border-t border-slate-800">
              <button
                onClick={() => setCurrentStep(3)}
                className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900 px-4 py-2.5 text-xs font-bold text-slate-300 hover:text-white"
              >
                <ArrowLeft className="h-4 w-4" /> Re-configure
              </button>
              <button
                onClick={handleTrain}
                disabled={loading}
                className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 px-8 py-3 text-xs font-black text-slate-950 shadow-xl shadow-cyan-500/25 hover:from-emerald-400 hover:to-cyan-400 transition"
              >
                {loading ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    <span>Executing Quantum Simulator ({configInfo.qubit_count} Qubits)...</span>
                  </>
                ) : (
                  <>
                    <Cpu className="h-4 w-4" />
                    <span>Train Classical KNN & Quantum KNN</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================= */}
      {/* STEP 5: RESULTS & SIDE-BY-SIDE COMPARISON                 */}
      {/* ========================================================= */}
      {currentStep === 5 && results && (
        <div className="space-y-6">
          {/* Top Score Comparison Cards */}
          <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
            {/* Quantum KNN Card */}
            <div className="rounded-2xl border-2 border-emerald-500/50 bg-gradient-to-br from-slate-900 via-emerald-950/30 to-slate-900 p-6 shadow-2xl">
              <div className="flex justify-between items-center border-b border-emerald-500/20 pb-3">
                <span className="rounded-full bg-emerald-500/20 px-3 py-0.5 text-xs font-bold text-emerald-300 border border-emerald-500/30 flex items-center gap-1.5">
                  <Cpu className="h-3.5 w-3.5" /> Quantum KNN (PennyLane)
                </span>
                <span className="text-xs font-mono text-emerald-400 font-bold">
                  {results.qubits_used} Qubits &bull; {results.feature_map.toUpperCase()} Map
                </span>
              </div>
              <div className="mt-4 grid grid-cols-2 gap-4">
                <div>
                  <span className="text-xs text-slate-400 block">Test Accuracy</span>
                  <h3 className="text-3xl font-black text-emerald-400 mt-1">
                    {(results.quantum_knn.accuracy * 100).toFixed(1)}%
                  </h3>
                </div>
                <div>
                  <span className="text-xs text-slate-400 block">Macro F1-Score</span>
                  <h3 className="text-3xl font-black text-white mt-1">
                    {(results.quantum_knn.f1_macro * 100).toFixed(1)}%
                  </h3>
                </div>
              </div>
              <div className="mt-4 grid grid-cols-3 gap-2 text-[11px] font-mono text-slate-300 border-t border-emerald-500/20 pt-3">
                <div>Precision: {(results.quantum_knn.precision_macro * 100).toFixed(1)}%</div>
                <div>Recall: {(results.quantum_knn.recall_macro * 100).toFixed(1)}%</div>
                <div>Inference: {results.quantum_knn.inference_time_sec}s</div>
              </div>
            </div>

            {/* Classical KNN Card */}
            <div className="rounded-2xl border-2 border-indigo-500/40 bg-gradient-to-br from-slate-900 via-indigo-950/30 to-slate-900 p-6 shadow-2xl">
              <div className="flex justify-between items-center border-b border-indigo-500/20 pb-3">
                <span className="rounded-full bg-indigo-500/20 px-3 py-0.5 text-xs font-bold text-indigo-300 border border-indigo-500/30 flex items-center gap-1.5">
                  <BarChart2 className="h-3.5 w-3.5" /> Classical KNN (Scikit-Learn)
                </span>
                <span className="text-xs font-mono text-indigo-400 font-bold">
                  Euclidean Metric &bull; k={results.k}
                </span>
              </div>
              <div className="mt-4 grid grid-cols-2 gap-4">
                <div>
                  <span className="text-xs text-slate-400 block">Test Accuracy</span>
                  <h3 className="text-3xl font-black text-indigo-400 mt-1">
                    {(results.classical_knn.accuracy * 100).toFixed(1)}%
                  </h3>
                </div>
                <div>
                  <span className="text-xs text-slate-400 block">Macro F1-Score</span>
                  <h3 className="text-3xl font-black text-white mt-1">
                    {(results.classical_knn.f1_macro * 100).toFixed(1)}%
                  </h3>
                </div>
              </div>
              <div className="mt-4 grid grid-cols-3 gap-2 text-[11px] font-mono text-slate-300 border-t border-indigo-500/20 pt-3">
                <div>Precision: {(results.classical_knn.precision_macro * 100).toFixed(1)}%</div>
                <div>Recall: {(results.classical_knn.recall_macro * 100).toFixed(1)}%</div>
                <div>Inference: {results.classical_knn.inference_time_sec}s</div>
              </div>
            </div>
          </div>

          {/* Performance Comparison Bar Chart */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 mb-3">
              Performance Benchmark Comparison
            </h3>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={[
                    { metric: 'Accuracy', Classical: results.classical_knn.accuracy * 100, Quantum: results.quantum_knn.accuracy * 100 },
                    { metric: 'Precision (Macro)', Classical: results.classical_knn.precision_macro * 100, Quantum: results.quantum_knn.precision_macro * 100 },
                    { metric: 'Recall (Macro)', Classical: results.classical_knn.recall_macro * 100, Quantum: results.quantum_knn.recall_macro * 100 },
                    { metric: 'F1-Score (Macro)', Classical: results.classical_knn.f1_macro * 100, Quantum: results.quantum_knn.f1_macro * 100 },
                  ]}
                >
                  <XAxis dataKey="metric" stroke="#94A3B8" fontSize={11} />
                  <YAxis domain={[0, 100]} stroke="#94A3B8" fontSize={11} unit="%" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#020617', borderColor: '#1E293B', borderRadius: '0.75rem' }}
                    formatter={(val) => [`${Number(val).toFixed(1)}%`, '']}
                  />
                  <Legend />
                  <Bar dataKey="Classical" fill="#6366F1" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Quantum" fill="#10B981" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Circuit Details */}
          {results.circuit_summary?.ascii_diagram && (
            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
              <div className="flex justify-between items-center border-b border-slate-800 pb-3 mb-3">
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-cyan-400" />
                  PennyLane Quantum Feature Map Circuit ({results.qubits_used} Qubits)
                </h3>
                <span className="text-xs font-mono text-cyan-400">
                  {results.circuit_summary.total_gates} Total Gates &bull; {results.circuit_summary.cnot_gates} CNOT Entanglers
                </span>
              </div>
              <div className="overflow-x-auto rounded-xl bg-slate-950 p-4 border border-slate-800">
                <pre className="font-mono text-xs text-cyan-300 leading-relaxed whitespace-pre">
                  {results.circuit_summary.ascii_diagram}
                </pre>
              </div>
            </div>
          )}

          <div className="flex justify-between items-center pt-2">
            <button
              onClick={() => setCurrentStep(4)}
              className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900 px-4 py-2.5 text-xs font-bold text-slate-300 hover:text-white"
            >
              <ArrowLeft className="h-4 w-4" /> Change Training Hyperparameters
            </button>
            <div className="flex gap-3">
              <button
                onClick={() => setCurrentStep(6)}
                className="flex items-center gap-2 rounded-xl bg-cyan-500 px-5 py-2.5 text-xs font-bold text-slate-950 shadow-lg shadow-cyan-500/25 hover:bg-cyan-400 transition"
              >
                Go to Live Prediction <ArrowRight className="h-4 w-4" />
              </button>
              <button
                onClick={() => setCurrentStep(7)}
                className="flex items-center gap-2 rounded-xl bg-indigo-600 px-5 py-2.5 text-xs font-bold text-white shadow-lg shadow-indigo-600/25 hover:bg-indigo-500 transition"
              >
                <Download className="h-4 w-4" /> Download PDF Report
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================= */}
      {/* STEP 6: LIVE PREDICTION (MANUAL & BATCH CSV)              */}
      {/* ========================================================= */}
      {currentStep === 6 && configInfo && (
        <div className="space-y-6">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-6 space-y-5">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-400 font-mono">
                Real-Time Inference Interface
              </span>
              <h3 className="text-lg font-black text-white mt-1">Classify New Custom Instances</h3>
              <p className="text-xs text-slate-300 mt-1">
                Enter values for the selected feature columns ({configInfo.selected_feature_names.join(', ')}). The system scales them and computes quantum state overlap fidelities against training samples.
              </p>
            </div>

            {/* Input Feature Fields */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {configInfo.selected_feature_names.map((feat) => {
                const stats = profile?.descriptive_statistics?.[feat] || { min: 0, max: 10, median: 5 };
                const currentVal = manualFeatures[feat] ?? stats.median;
                return (
                  <div key={feat} className="rounded-xl bg-slate-950 p-3.5 border border-slate-800">
                    <div className="flex justify-between items-center text-xs mb-2">
                      <span className="font-bold text-slate-200 capitalize">{feat.replace(/_/g, ' ')}</span>
                      <input
                        type="number"
                        step="0.1"
                        value={currentVal}
                        onChange={(e) =>
                          setManualFeatures({ ...manualFeatures, [feat]: parseFloat(e.target.value) || 0 })
                        }
                        className="w-16 rounded-md bg-slate-900 border border-slate-700 px-1.5 py-0.5 text-right font-mono text-xs font-bold text-cyan-400 focus:outline-none"
                      />
                    </div>
                    <input
                      type="range"
                      min={stats.min ?? 0}
                      max={stats.max ?? 10}
                      step={((stats.max ?? 10) - (stats.min ?? 0)) / 100 || 0.1}
                      value={currentVal}
                      onChange={(e) =>
                        setManualFeatures({ ...manualFeatures, [feat]: parseFloat(e.target.value) })
                      }
                      className="w-full accent-cyan-400 cursor-pointer"
                    />
                  </div>
                );
              })}
            </div>

            <div className="flex justify-center pt-2">
              <button
                onClick={handlePredict}
                disabled={loading}
                className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 px-8 py-3 text-xs font-black text-slate-950 shadow-xl shadow-cyan-500/25 hover:from-emerald-400 hover:to-cyan-400 transition"
              >
                {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <PlayCircle className="h-4 w-4" />}
                Run Live Quantum & Classical Prediction
              </button>
            </div>

            {/* Prediction Output */}
            {prediction && (
              <div className="rounded-2xl border border-cyan-500/40 bg-gradient-to-br from-slate-900 via-cyan-950/20 to-slate-900 p-6 space-y-4">
                <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                  <div>
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Predicted Class</span>
                    <h3 className="text-2xl font-black text-emerald-400 mt-0.5 capitalize">
                      {prediction.quantum_predicted_class}
                    </h3>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block">Classical Match</span>
                    <span className={`text-sm font-bold ${prediction.agreement ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {prediction.agreement ? '✓ 100% Agreement' : `Classical: ${prediction.classical_predicted_class}`}
                    </span>
                  </div>
                </div>

                <p className="text-xs text-slate-300 font-medium">
                  {prediction.explanation}
                </p>

                {/* Nearest Neighbors Table */}
                {prediction.nearest_neighbors?.length > 0 && (
                  <div className="overflow-x-auto pt-2">
                    <table className="w-full text-left text-xs">
                      <thead>
                        <tr className="border-b border-slate-800 text-[11px] font-bold uppercase text-slate-400 font-mono">
                          <th className="py-2 px-3">Rank</th>
                          <th className="py-2 px-3">Train Index</th>
                          <th className="py-2 px-3">Class</th>
                          <th className="py-2 px-3">State Fidelity (F)</th>
                          <th className="py-2 px-3">Quantum Distance (1 - F)</th>
                          <th className="py-2 px-3">Voting Influence</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 font-mono">
                        {prediction.nearest_neighbors.map((n) => (
                          <tr key={n.rank} className="hover:bg-slate-850/50">
                            <td className="py-2.5 px-3 font-bold text-cyan-400">#{n.rank}</td>
                            <td className="py-2.5 px-3 text-slate-300">Index {n.train_index}</td>
                            <td className="py-2.5 px-3 text-white font-sans font-bold capitalize">{n.class_label}</td>
                            <td className="py-2.5 px-3 font-bold text-emerald-400">
                              {n.fidelity !== null ? `${(n.fidelity * 100).toFixed(2)}%` : '--'}
                            </td>
                            <td className="py-2.5 px-3 text-amber-400">{n.quantum_distance ?? '--'}</td>
                            <td className="py-2.5 px-3 text-indigo-300">
                              {n.voting_weight ? `${(n.voting_weight * 100).toFixed(1)}%` : '--'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ========================================================= */}
      {/* STEP 7: DOWNLOAD PDF RESEARCH REPORT                      */}
      {/* ========================================================= */}
      {currentStep === 7 && datasetId && (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-8 text-center space-y-4">
          <FileText className="h-16 w-16 text-cyan-400 mx-auto" />
          <h3 className="text-xl font-black text-white">Download Scientific Research Report</h3>
          <p className="text-xs text-slate-300 max-w-lg mx-auto leading-relaxed">
            Generate an official, publication-quality PDF report containing dataset information, statistical distributions, preprocessing parameters, PennyLane circuit properties, and comprehensive Classical vs Quantum KNN metrics.
          </p>

          <div className="pt-2">
            <a
              href={api.getReportDownloadUrl(datasetId)}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 px-8 py-3.5 text-xs font-black text-white shadow-xl shadow-cyan-500/25 hover:from-cyan-400 hover:to-indigo-500 transition cursor-pointer"
            >
              <Download className="h-4 w-4" />
              Download Official PDF Research Report
            </a>
          </div>
        </div>
      )}
    </div>
  );
}

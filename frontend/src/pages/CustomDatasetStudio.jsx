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
  const [downloadingReport, setDownloadingReport] = useState(false);
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
      const detail = err.response?.data?.message || err.message;
      setErrorMsg(`Upload failed: ${detail}. If connecting to local server, ensure 'python backend/app.py' is running on port 5000.`);
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
      const detail = err.response?.data?.message || err.message;
      setErrorMsg(`Failed to load sample dataset '${sampleName}': ${detail}. If connecting to local server, ensure 'python backend/app.py' is running on port 5000.`);
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

  const handleDownloadReport = async () => {
    if (!datasetId) return;
    setDownloadingReport(true);
    setErrorMsg('');
    try {
      const res = await api.downloadReportBlob(datasetId);
      const blob = new Blob([res.data], { type: 'application/pdf' });
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `quantum_knn_research_report_${datasetId}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => window.URL.revokeObjectURL(downloadUrl), 5000);
    } catch (err) {
      console.warn('Blob download failed, falling back to direct URL:', err);
      try {
        const directUrl = api.getReportDownloadUrl(datasetId);
        window.open(directUrl, '_blank');
      } catch (e) {
        setErrorMsg('Failed to download PDF report. Ensure model is trained first.');
      }
    } finally {
      setDownloadingReport(false);
    }
  };

  const getDatasetContext = (filename = '', targetCol = '', classes = [], overview = null, profile = null) => {
    const name = (filename || '').toLowerCase();
    const targetCounts = profile?.target_distribution || {};
    const totalCount = Object.values(targetCounts).reduce((a, b) => a + b, 0) || overview?.total_rows || 1;

    // Helper to format class breakdown
    const classBreakdown = classes.map((c) => {
      const count = targetCounts[c] ?? '--';
      const pct = typeof count === 'number' ? ((count / totalCount) * 100).toFixed(1) + '%' : '';
      return { className: c, count, pct };
    });

    if (name.includes('student')) {
      return {
        domain: 'Education & Academic Analytics',
        badge: 'Academic Research',
        title: 'Student Academic Grade Tier Classification',
        summary: 'This dataset investigates the multi-factorial relationship between weekly study habits, class attendance rate, prior examination scores, tutoring sessions, parental guidance, and sleep patterns to predict final academic achievement tiers.',
        dataCollection: 'Compiled from university Learning Management Systems (LMS) and institutional student records, tracking lecture attendance, weekly study logs, and examination grades across an undergraduate student cohort.',
        targetObjective: `Classifies each student into one of ${classes.length} achievement tiers (${classes.join(', ')}). In academic institutions, early identification of student tiers enables targeted pedagogical interventions, remedial tutoring, and academic counseling before final semester examinations.`,
        practicalImpact: 'Enables academic departments to proactively reduce course failure and dropout rates by up to 25%, optimize faculty tutoring office hours, and nominate top-tier scholars for academic awards.',
        decisionGuidelines: 'Distinction: recommend for research seminars & honors mentoring; Merit: maintain current academic pace; Pass / At-Risk: trigger mandatory tutoring and weekly attendance monitoring.',
        classBreakdown,
        featureDictionary: [
          { name: 'study_hours_weekly', label: 'Weekly Study Hours', unit: 'Hours / week', type: 'Continuous', range: '5.0 - 42.0 hrs', description: 'Active self-study and revision time outside scheduled classes. Strongest positive predictor of higher academic grade tiers.' },
          { name: 'attendance_percentage', label: 'Lecture Attendance', unit: '% Attendance', type: 'Continuous', range: '50.0% - 100.0%', description: 'Lecture and laboratory presence rate. Sub-75% attendance triggers institutional academic probation warnings.' },
          { name: 'previous_grade_score', label: 'Prior Exam Score', unit: 'Marks (0 - 100)', type: 'Continuous', range: '40.0 - 98.0 marks', description: 'Baseline academic capability measured from past semester standardized examinations or entrance GPA.' },
          { name: 'tutoring_sessions', label: 'Tutoring Mentorship', unit: 'Sessions / month', type: 'Discrete Count', range: '0 - 8 sessions', description: 'Supplemental peer tutoring, teaching assistant consultations, or remedial problem-solving workshops attended.' },
          { name: 'parental_involvement', label: 'Guardian Oversight', unit: 'Tier (Low / Med / High)', type: 'Ordinal Categorical', range: '3 discrete levels', description: 'Level of home academic encouragement, parent-teacher engagement, and study environment monitoring.' },
          { name: 'sleep_hours_daily', label: 'Nightly Sleep Duration', unit: 'Hours / day', type: 'Continuous', range: '4.5 - 9.0 hrs', description: 'Average daily sleep duration reflecting cognitive rest, stress management, and memory consolidation.' },
        ],
        featureInsights: 'Key drivers such as weekly study hours and attendance percentage exhibit multi-factor interactions that are mapped onto multi-qubit Pauli rotation states.',
      };
    } else if (name.includes('agri') || name.includes('crop')) {
      return {
        domain: 'Smart Agriculture & Soil Science',
        badge: 'Agritech Research',
        title: 'Precision Soil Chemistry & Atmospheric Crop Recommendation',
        summary: 'This agricultural dataset examines multi-dimensional soil chemistry (Nitrogen, Phosphorus, Potassium ratios, pH) and ambient meteorological parameters (temperature, humidity, rainfall) to recommend the optimal crop variety for sustainable farm yields.',
        dataCollection: 'Collected using IoT-enabled agricultural field telemetry stations, precision soil chemical test kits, and regional meteorological satellite precipitation feeds across varied agro-climatic farming zones.',
        targetObjective: `Recommends the best-suited crop among ${classes.length} agricultural varieties (${classes.join(', ')}). Accurate crop recommendation prevents total crop failure, optimizes chemical fertilizer expenditure, and conserves soil microbial health.`,
        practicalImpact: 'Helps agricultural cooperatives and smallholder farmers maximize harvest success, reduce synthetic fertilizer over-application by up to 30%, conserve irrigation groundwater, and boost farm profitability.',
        decisionGuidelines: 'Rice: requires high seasonal rainfall (>180mm) and high relative humidity; Maize: moderate nitrogen and temperate conditions; Cotton: high potassium and warm temperature; Chickpea: low moisture and neutral-to-basic pH.',
        classBreakdown,
        featureDictionary: [
          { name: 'nitrogen_ratio', label: 'Soil Nitrogen (N)', unit: 'kg / hectare (kg/ha)', type: 'Continuous', range: '10.0 - 140.0 kg/ha', description: 'Essential macronutrient ratio responsible for leaf chlorophyll synthesis, vegetative biomass, and vegetative vigor.' },
          { name: 'phosphorus_ratio', label: 'Soil Phosphorus (P)', unit: 'kg / hectare (kg/ha)', type: 'Continuous', range: '5.0 - 145.0 kg/ha', description: 'Crucial for root proliferation, early seedling vigor, flower initiation, and crop reproductive energy storage.' },
          { name: 'potassium_ratio', label: 'Soil Potassium (K)', unit: 'kg / hectare (kg/ha)', type: 'Continuous', range: '10.0 - 205.0 kg/ha', description: 'Regulates plant stomatal conductance, osmotic water balance, disease resistance, and grain starch accumulation.' },
          { name: 'temperature_celsius', label: 'Ambient Temperature', unit: 'Degrees Celsius (°C)', type: 'Continuous', range: '15.0°C - 42.0°C', description: 'Field thermal conditions governing crop metabolic rates, flowering times, and heat stress thresholds.' },
          { name: 'humidity_percentage', label: 'Relative Humidity', unit: '% Atmospheric RH', type: 'Continuous', range: '20.0% - 99.0%', description: 'Vapor pressure deficit governing plant transpiration and susceptibility to foliar fungal pathogens.' },
          { name: 'soil_ph', label: 'Soil pH Chemistry', unit: 'pH Scale (0 - 14)', type: 'Continuous', range: '4.5 - 8.5 pH', description: 'Soil acidity or alkalinity index determining chemical mineral solubility and macro-nutrient root absorption.' },
          { name: 'rainfall_mm', label: 'Seasonal Precipitation', unit: 'Millimeters (mm)', type: 'Continuous', range: '30.0 - 300.0 mm', description: 'Cumulative seasonal rainfall determining groundwater availability and irrigation necessity.' },
        ],
        featureInsights: 'Soil nutrient ratios (N-P-K) and environmental humidity create intricate non-linear boundaries where quantum entanglement gates (CNOT) model co-dependencies between soil pH and rainfall.',
      };
    } else if (name.includes('social') || name.includes('ad') || name.includes('consumer')) {
      return {
        domain: 'Consumer Behavior & Digital Marketing',
        badge: 'Marketing Intelligence',
        title: 'Social Network Ad Conversion & Purchase Decision',
        summary: 'This dataset explores consumer demographics, estimated household salaries, daily active screen time, and weekly platform engagement to predict digital advertisement conversion likelihood.',
        dataCollection: 'Gathered from digital advertising platform clickstreams and e-commerce shopping sessions, capturing verified demographic profiles, salary estimates, and in-app interaction telemetry.',
        targetObjective: `Determines user purchase conversion actions (${classes.join(' vs ')}). High precision prevents wasted advertising ad spend by focusing promotional campaigns on receptive consumer profiles.`,
        practicalImpact: 'Reduces Customer Acquisition Cost (CAC) by up to 35%, eliminates wasted ad budget on non-converting demographics, and enables automated programmatic real-time ad bidding.',
        decisionGuidelines: 'Purchased: serve targeted product checkout incentives, dynamic remarketing, and loyalty promotions; Not Purchased: avoid high-cost bid auctions to prevent ad fatigue and budget leakage.',
        classBreakdown,
        featureDictionary: [
          { name: 'user_age', label: 'Consumer Age', unit: 'Years', type: 'Continuous', range: '18 - 65 yrs', description: 'Chronological age of the user. Older demographic cohorts frequently exhibit higher disposable income conversion propensity.' },
          { name: 'estimated_annual_salary', label: 'Estimated Annual Salary', unit: 'USD ($) / year', type: 'Continuous', range: '$15,000 - $150,000', description: 'Estimated gross yearly household income determining disposable purchasing power and luxury product affordability.' },
          { name: 'daily_social_hours', label: 'Daily Social Screen Time', unit: 'Hours / day', type: 'Continuous', range: '0.5 - 7.5 hrs', description: 'Active hours spent browsing social media feeds daily, reflecting digital attention span and ad exposure opportunity.' },
          { name: 'weekly_interactions', label: 'Weekly Platform Engagements', unit: 'Actions / week', type: 'Discrete Count', range: '5 - 95 actions', description: 'Aggregated weekly frequency of likes, comments, bookmarking, and link clicks indicating platform engagement level.' },
          { name: 'primary_device', label: 'Client Device Category', unit: 'Device Type', type: 'Categorical', range: 'Mobile, Desktop, Tablet', description: 'Dominant hardware platform utilized during browsing, influencing checkout friction and payment UX.' },
        ],
        featureInsights: 'The interplay between user age and disposable annual salary is mapped into non-linear Hilbert space angles, distinguishing spontaneous purchasers from non-converting users.',
      };
    } else if (name.includes('iris')) {
      return {
        domain: 'Botanical Morphometrics & Taxonomy',
        badge: 'Pattern Recognition',
        title: 'Fisher Iris Morphological Species Classification',
        summary: 'The classic Fisher Iris benchmark evaluates sepal length, sepal width, petal length, and petal width to discriminate between three morphologically distinct Iris species.',
        dataCollection: 'Collected by botanist Edgar Anderson on the Gaspé Peninsula (Quebec, Canada), systematically recording 150 specimens across three species from wild colonies.',
        targetObjective: `Classifies botanical specimens into species (${classes.join(', ')}). Iris Setosa is linearly separable, while Versicolor and Virginica share overlapping geometric boundaries in classical Euclidean space.`,
        practicalImpact: 'Serves as the golden standard benchmark for evaluating non-linear classifier boundaries, geometric kernel mappings, and quantum state Hilbert space separability.',
        decisionGuidelines: 'Setosa: completely isolated by petal dimensions (<2cm); Versicolor & Virginica: require joint sepal and petal multi-qubit entanglement to resolve boundary samples.',
        classBreakdown,
        featureDictionary: [
          { name: 'sepal_length', label: 'Sepal Length', unit: 'Centimeters (cm)', type: 'Continuous', range: '4.3 - 7.9 cm', description: 'Longitudinal length of the outer floral calyx sepal protecting the flower bud.' },
          { name: 'sepal_width', label: 'Sepal Width', unit: 'Centimeters (cm)', type: 'Continuous', range: '2.0 - 4.4 cm', description: 'Transverse width of the outer floral calyx sepal.' },
          { name: 'petal_length', label: 'Petal Length', unit: 'Centimeters (cm)', type: 'Continuous', range: '1.0 - 6.9 cm', description: 'Longitudinal length of the inner floral corolla petal. Most powerful morphological discriminator.' },
          { name: 'petal_width', label: 'Petal Width', unit: 'Centimeters (cm)', type: 'Continuous', range: '0.1 - 2.5 cm', description: 'Transverse width of the inner floral corolla petal.' },
        ],
        featureInsights: 'Petal length and petal width dimensions serve as the strongest discriminators, transformed into 4-qubit quantum rotation angles.',
      };
    } else if (name.includes('wine')) {
      return {
        domain: 'Chemical Spectrometry & Enology',
        badge: 'Chemical Analytics',
        title: 'Wine Cultivar Spectrometric Classification',
        summary: 'Derived from chemical analysis of wines grown in the same region in Italy, measuring 13 constituents (alcohol, malic acid, ash, flavanoids, color intensity, proline).',
        dataCollection: 'Derived from high-performance liquid chromatography (HPLC) and atomic absorption spectrometry conducted on wines harvested in the Piedmont region of Italy.',
        targetObjective: `Distinguishes wine origin across ${classes.length} cultivars (${classes.join(', ')}). High cultivar accuracy protects geographical indicator designations and vintage authenticity.`,
        practicalImpact: 'Chemically authenticates Protected Designation of Origin (PDO) vintages, prevents commercial counterfeiting and dilution fraud, and guides enologists in fermentation quality control.',
        decisionGuidelines: 'Cultivar 1: high alcohol and high proline (>1000 mg/L); Cultivar 2: lower flavanoids and lighter color intensity; Cultivar 3: elevated malic acid and high ash alcalinity.',
        classBreakdown,
        featureDictionary: [
          { name: 'alcohol', label: 'Alcohol Concentration', unit: '% by Volume', type: 'Continuous', range: '11.0% - 14.8%', description: 'Ethanol concentration synthesized during yeast fermentation of grape sugars.' },
          { name: 'malic_acid', label: 'Malic Acid', unit: 'g / L', type: 'Continuous', range: '0.74 - 5.80 g/L', description: 'Dicarboxylic organic acid providing crisp tartness; diminishes with malolactic fermentation.' },
          { name: 'ash', label: 'Ash Mineral Residue', unit: 'g / L', type: 'Continuous', range: '1.36 - 3.23 g/L', description: 'Total inorganic mineral residue remaining after complete organic combustion.' },
          { name: 'alcalinity_of_ash', label: 'Ash Alcalinity', unit: 'mEq / L', type: 'Continuous', range: '10.6 - 30.0', description: 'Buffering capacity of mineral ash against organic wine acids.' },
          { name: 'flavanoids', label: 'Flavanoids Polyphenols', unit: 'g / L', type: 'Continuous', range: '0.34 - 5.08 g/L', description: 'Potent antioxidant polyphenols responsible for color stability, astringency, and wine aging potential.' },
          { name: 'color_intensity', label: 'Color Intensity', unit: 'Absorbance Units', type: 'Continuous', range: '1.28 - 13.0', description: 'Spectrophotometric absorbance at 420 nm + 520 nm reflecting anthocyanin extraction.' },
          { name: 'proline', label: 'Proline Amino Acid', unit: 'mg / L', type: 'Continuous', range: '278 - 1680 mg/L', description: 'Dominant amino acid marker that cannot be assimilated by yeast, reflecting grape ripeness and cultivar origin.' },
        ],
        featureInsights: 'Chemical concentrations (flavanoids, color intensity, proline) are reduced to optimal qubit representations to detect subtle cultivar differences.',
      };
    } else if (name.includes('liver')) {
      return {
        domain: 'Clinical Gastroenterology & Healthcare',
        badge: 'Medical Diagnostics',
        title: 'Liver Disease Patient Diagnosis & Risk Assessment',
        summary: 'Collected in North East Andhra Pradesh, India, evaluating age, gender, bilirubin levels, alkaline phosphatase, and protein ratios to detect patient liver disease risk.',
        dataCollection: 'Gathered from clinical patient health records and biochemical laboratory blood panels across diagnostic hospitals in Andhra Pradesh, India.',
        targetObjective: `Classifies clinical outcomes into diseased vs healthy liver profiles (${classes.join(' vs ')}). High sensitivity (recall) is clinically critical to avoid missing at-risk patients requiring intervention.`,
        practicalImpact: 'Supports rapid medical triaging in outpatient gastroenterology clinics, alerting physicians to patients with elevated hepatic enzyme necrosis who require ultrasound imaging.',
        decisionGuidelines: 'Liver Disease: immediate hepatology specialist referral, liver function panel monitoring, and ultrasound elastography; Healthy Control: routine annual health screening.',
        classBreakdown,
        featureDictionary: [
          { name: 'Age', label: 'Patient Age', unit: 'Years', type: 'Continuous', range: '4 - 90 yrs', description: 'Patient chronological age. Incidence of chronic liver injury increases in middle-to-older age demographics.' },
          { name: 'Total_Bilirubin', label: 'Total Serum Bilirubin', unit: 'mg / dL', type: 'Continuous', range: '0.4 - 75.0 mg/dL', description: 'Orange-yellow pigment breakdown product of hemoglobin. Elevated bilirubin produces clinical jaundice.' },
          { name: 'Direct_Bilirubin', label: 'Direct Bilirubin', unit: 'mg / dL', type: 'Continuous', range: '0.1 - 19.7 mg/dL', description: 'Conjugated water-soluble bilirubin processed by functional liver hepatocytes.' },
          { name: 'Alkaline_Phosphotase', label: 'Alkaline Phosphatase (ALP)', unit: 'IU / L', type: 'Continuous', range: '63 - 2110 IU/L', description: 'Enzyme concentrated in biliary duct cells; elevated ALP indicates cholestasis or bile duct obstruction.' },
          { name: 'Alamine_Aminotransferase', label: 'ALT Transaminase', unit: 'IU / L', type: 'Continuous', range: '10 - 2000 IU/L', description: 'Primary biomarker enzyme specific to liver parenchyma necrosis and viral hepatitis injury.' },
          { name: 'Aspartate_Aminotransferase', label: 'AST Transaminase', unit: 'IU / L', type: 'Continuous', range: '10 - 4929 IU/L', description: 'Mitochondrial enzyme released into bloodstream upon hepatocellular lysis or myocardial tissue damage.' },
          { name: 'Albumin', label: 'Serum Albumin', unit: 'g / dL', type: 'Continuous', range: '0.9 - 5.5 g/dL', description: 'Major blood plasma protein synthesized exclusively by the liver; depleted in chronic hepatic cirrhosis.' },
        ],
        featureInsights: 'Enzyme markers (ALT, AST) and bilirubin levels exhibit severe variance and skewness, normalized into bounded quantum rotation angles.',
      };
    } else {
      // Dynamic profile for custom uploaded datasets
      const cols = overview?.column_names?.filter((c) => c !== targetCol) || [];
      const dynamicDict = cols.slice(0, 10).map((col) => {
        const stats = profile?.descriptive_statistics?.[col];
        const isNum = stats?.mean !== null && stats?.mean !== undefined;
        let unit = 'Standardized Unit [AU]';
        const colLow = col.toLowerCase();
        if (colLow.includes('age')) unit = 'Years';
        else if (colLow.includes('salary') || colLow.includes('income') || colLow.includes('price') || colLow.includes('cost') || colLow.includes('revenue')) unit = 'USD ($)';
        else if (colLow.includes('pct') || colLow.includes('percent') || colLow.includes('rate') || colLow.includes('ratio')) unit = '% / Ratio';
        else if (colLow.includes('hour') || colLow.includes('time') || colLow.includes('duration')) unit = 'Hours';
        else if (colLow.includes('count') || colLow.includes('num') || colLow.includes('freq')) unit = 'Integer Count';
        else if (colLow.includes('score') || colLow.includes('grade') || colLow.includes('mark')) unit = 'Score Points';

        const rangeStr = isNum && stats ? `${stats.min} to ${stats.max}` : `${stats?.unique_count || '--'} unique levels`;
        return {
          name: col,
          label: col.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
          unit,
          type: isNum ? 'Continuous Numerical' : 'Categorical Attribute',
          range: rangeStr,
          description: `Feature '${col}' normalized and mapped into quantum rotation angle θ = 2·arctan(x) in PennyLane circuit.`,
        };
      });

      return {
        domain: 'Custom Tabular Machine Learning',
        badge: 'User Ingested Data',
        title: `${filename || 'Custom'} Supervised Classification Task`,
        summary: `This uploaded dataset comprises ${classes.length} target classes evaluated across multi-attribute features using both Classical KNN and PennyLane Quantum KNN.`,
        dataCollection: `Uploaded directly by user (${overview?.filename || 'dataset'}) containing ${overview?.total_rows || '--'} rows and ${overview?.total_columns || '--'} columns.`,
        targetObjective: `The classification pipeline predicts target attribute '${targetCol}' (${classes.join(', ')}). Evaluating state overlap fidelity provides a quantum-mechanic alternative to Euclidean distance.`,
        practicalImpact: 'Evaluates whether parameterized quantum Hilbert state space embedding offers separability advantages for domain-specific tabular datasets.',
        decisionGuidelines: `Samples are assigned to one of ${classes.length} categories based on distance-weighted voting from k nearest training neighbors.`,
        classBreakdown,
        featureDictionary: dynamicDict,
        featureInsights: 'Continuous numerical attributes are standardized and mapped into unitary quantum rotation angles to explore non-Euclidean nearest-neighbor margins.',
      };
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

                <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
                  {[
                    { id: 'iris', name: 'Iris Flower', rows: '150 rows', cols: '4 features', badge: 'Botany' },
                    { id: 'student', name: 'Student Grades', rows: '150 rows', cols: '6 features', badge: 'Education' },
                    { id: 'agriculture', name: 'Crop Harvest', rows: '160 rows', cols: '7 features', badge: 'Agriculture' },
                    { id: 'social', name: 'Social Ads Action', rows: '180 rows', cols: '5 features', badge: 'Social Media' },
                    { id: 'wine', name: 'Wine Dataset', rows: '178 rows', cols: '13 features', badge: 'Chemistry' },
                    { id: 'indian_liver', name: 'Indian Liver Patient', rows: '583 rows', cols: '10 features', badge: 'Healthcare' },
                  ].map((s) => (
                    <button
                      key={s.id}
                      onClick={() => handleLoadSample(s.id)}
                      disabled={loading}
                      className="rounded-xl border border-slate-800 bg-slate-950 p-3.5 text-left transition hover:border-cyan-400 hover:bg-slate-850 cursor-pointer flex flex-col justify-between"
                    >
                      <div>
                        <div className="flex items-center justify-between">
                          <span className="text-[9px] font-bold uppercase tracking-wider text-cyan-400 bg-cyan-950/60 px-1.5 py-0.5 rounded border border-cyan-800/40">
                            {s.badge}
                          </span>
                        </div>
                        <div className="font-bold text-xs text-white capitalize mt-1.5">{s.name}</div>
                        <div className="text-[11px] font-mono text-slate-400 mt-0.5">{s.rows} &bull; {s.cols}</div>
                      </div>
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

          {/* Dataset Domain Context, Real-World Impact & Attribute Dictionary */}
          {(() => {
            const detectedTarget = targetColumn || profile.recommended_target || profile.candidate_target_columns?.[0] || '';
            const detectedClasses = profile.target_distribution ? Object.keys(profile.target_distribution) : [];
            const dsInfo = getDatasetContext(overview.filename, detectedTarget, detectedClasses, overview, profile);

            return (
              <div className="rounded-2xl border border-indigo-500/30 bg-gradient-to-br from-slate-900 via-slate-900/90 to-slate-950 p-6 space-y-5">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <span className="rounded-full bg-cyan-500/20 px-3 py-1 text-xs font-bold uppercase tracking-wider text-cyan-300 border border-cyan-500/30 flex items-center gap-1.5">
                      <Database className="h-3.5 w-3.5" /> {dsInfo.domain}
                    </span>
                    <span className="rounded-full bg-indigo-500/20 px-2.5 py-1 text-xs font-mono text-indigo-300 border border-indigo-500/30">
                      {dsInfo.badge}
                    </span>
                  </div>
                  <div className="text-xs text-slate-400 font-mono">
                    Target Column: <strong className="text-emerald-400">{detectedTarget || 'Supervised Target'}</strong>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-black text-white">{dsInfo.title}</h3>
                  <p className="text-xs text-slate-300 mt-1.5 leading-relaxed">{dsInfo.summary}</p>
                </div>

                {/* 3-Column Context Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5 text-xs">
                  {/* Origin */}
                  <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-1.5">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-400 font-mono block">
                      Data Collection Context & Origin
                    </span>
                    <p className="text-slate-300 leading-relaxed text-[11px]">{dsInfo.dataCollection}</p>
                  </div>

                  {/* Objective & Classes */}
                  <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-1.5">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400 font-mono block">
                      Target Classification Objective
                    </span>
                    <p className="text-slate-300 leading-relaxed text-[11px]">{dsInfo.targetObjective}</p>
                    {dsInfo.classBreakdown?.length > 0 && (
                      <div className="pt-2">
                        <span className="text-[10px] text-slate-400 block mb-1">Class Distribution Breakdown:</span>
                        <div className="flex flex-wrap gap-1.5">
                          {dsInfo.classBreakdown.map((cb) => (
                            <span key={cb.className} className="rounded-md bg-emerald-950/60 border border-emerald-800/40 px-2 py-0.5 text-[10px] font-mono text-emerald-300">
                              {cb.className}: <strong>{cb.count}</strong> {cb.pct && `(${cb.pct})`}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Impact & Practical Guidelines */}
                  <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-1.5">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-400 font-mono block">
                      Real-World Impact & Actionable Guidance
                    </span>
                    <p className="text-slate-300 leading-relaxed text-[11px]">{dsInfo.practicalImpact}</p>
                    <div className="pt-2 text-[10px] text-indigo-300 border-t border-slate-800/80 font-mono">
                      <strong>Decision Rule:</strong> {dsInfo.decisionGuidelines}
                    </div>
                  </div>
                </div>

                {/* Feature Attribute Dictionary & Real-World Units Table */}
                {dsInfo.featureDictionary?.length > 0 && (
                  <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-3">
                    <div className="flex justify-between items-center border-b border-slate-800 pb-2">
                      <h4 className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
                        <Sliders className="h-3.5 w-3.5" /> Feature Attribute Dictionary & Measurement Units
                      </h4>
                      <span className="text-[10px] text-slate-400 font-mono">
                        {dsInfo.featureDictionary.length} Documented Attributes
                      </span>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead>
                          <tr className="border-b border-slate-800 text-[10px] font-bold uppercase tracking-wider text-slate-400 font-mono">
                            <th className="py-2 px-3">Attribute Name & Label</th>
                            <th className="py-2 px-3">Physical Unit</th>
                            <th className="py-2 px-3">Data Type</th>
                            <th className="py-2 px-3">Observed Range</th>
                            <th className="py-2 px-3">Domain Significance & Quantum Mapping</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/60 font-mono">
                          {dsInfo.featureDictionary.map((feat) => (
                            <tr key={feat.name} className="hover:bg-slate-900/60 transition">
                              <td className="py-2.5 px-3">
                                <div className="font-bold text-white font-sans">{feat.label}</div>
                                <div className="text-[10px] text-slate-400 font-mono">{feat.name}</div>
                              </td>
                              <td className="py-2.5 px-3">
                                <span className="rounded bg-cyan-950/80 px-2 py-0.5 text-[10px] font-bold text-cyan-300 border border-cyan-800/40">
                                  {feat.unit}
                                </span>
                              </td>
                              <td className="py-2.5 px-3 text-indigo-300 text-[11px] font-sans">{feat.type}</td>
                              <td className="py-2.5 px-3 text-emerald-400 text-[11px]">{feat.range}</td>
                              <td className="py-2.5 px-3 text-slate-300 text-[11px] font-sans max-w-xs">{feat.description}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            );
          })()}

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
      {currentStep === 5 && results && (() => {
        const dsContext = getDatasetContext(
          overview?.filename,
          results.target_column,
          results.classes,
          overview,
          profile
        );
        const accuracyDelta = (results.performance_comparison?.accuracy_delta ?? 0) * 100;
        const higherModel = results.performance_comparison?.higher_accuracy_model || 'Evaluated';

        return (
          <div className="space-y-6">
            {/* Domain Context & Dataset Overview Card */}
            <div className="rounded-2xl border border-cyan-500/30 bg-gradient-to-br from-slate-900 via-indigo-950/30 to-slate-900 p-6 shadow-xl space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                  <span className="rounded-full bg-cyan-500/20 px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-cyan-300 border border-cyan-500/30">
                    {dsContext.domain}
                  </span>
                  <span className="rounded-full bg-indigo-500/20 px-2.5 py-1 text-[10px] font-mono text-indigo-300 border border-indigo-500/30">
                    {dsContext.badge}
                  </span>
                </div>
                <div className="text-xs text-slate-400 font-mono">
                  File: <strong className="text-white">{overview?.filename || datasetId}</strong> &bull; {overview?.total_rows} Rows &bull; {overview?.total_columns} Features
                </div>
              </div>

              <div>
                <h3 className="text-lg font-black text-white flex items-center gap-2">
                  <Database className="h-5 w-5 text-cyan-400" />
                  {dsContext.title}
                </h3>
                <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                  {dsContext.summary}
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1">
                {/* Collection Origin */}
                <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3.5 space-y-1">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-400 font-mono block">
                    Data Collection Origin
                  </span>
                  <p className="text-xs text-slate-300 leading-relaxed text-[11px]">
                    {dsContext.dataCollection}
                  </p>
                </div>

                {/* Target Objective & Classes */}
                <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3.5 space-y-1">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400 font-mono block">
                    Target Objective & Meaning
                  </span>
                  <p className="text-xs text-slate-300 leading-relaxed text-[11px]">
                    {dsContext.targetObjective}
                  </p>
                  {dsContext.classBreakdown?.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 pt-1.5">
                      {dsContext.classBreakdown.map((cb) => (
                        <span key={cb.className} className="rounded-md bg-emerald-950/60 border border-emerald-800/40 px-2 py-0.5 text-[10px] font-mono text-emerald-300">
                          {cb.className}: <strong>{cb.count}</strong> {cb.pct && `(${cb.pct})`}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Real-World Impact & Decision Support */}
                <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3.5 space-y-1">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-400 font-mono block">
                    Real-World Impact & Decisions
                  </span>
                  <p className="text-xs text-slate-300 leading-relaxed text-[11px]">
                    {dsContext.practicalImpact}
                  </p>
                  <div className="text-[10px] text-indigo-300 pt-1 font-mono border-t border-slate-800/60">
                    Rule: {dsContext.decisionGuidelines}
                  </div>
                </div>
              </div>

              {/* Selected Quantum Features & Physical Units Grid */}
              <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3.5 space-y-2">
                <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-400 font-mono block">
                  Quantum Hilbert Space Features & Real-World Units ({results.qubits_used} Qubits Allocated)
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
                  {results.selected_features?.map((fName, idx) => {
                    const featInfo = dsContext.featureDictionary?.find((fd) => fd.name === fName) || {
                      label: fName,
                      unit: 'Standardized Unit [AU]',
                      type: 'Continuous',
                      description: 'Feature normalized and mapped to Pauli rotation angle.'
                    };
                    return (
                      <div key={fName} className="rounded-lg bg-slate-900 border border-slate-800 p-2.5 text-xs space-y-1">
                        <div className="flex justify-between items-start">
                          <strong className="text-white font-sans text-[11px] truncate block max-w-[120px]">{featInfo.label}</strong>
                          <span className="rounded bg-cyan-950 px-1.5 py-0.5 text-[9px] font-mono font-bold text-cyan-300 border border-cyan-800/40">
                            {featInfo.unit}
                          </span>
                        </div>
                        <div className="text-[10px] text-indigo-400 font-mono">
                          Wire q[{idx}] &bull; &theta; = 2&middot;arctan(x)
                        </div>
                        <div className="text-[10px] text-slate-400 leading-tight font-sans line-clamp-2">
                          {featInfo.description}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

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

            {/* Descriptive Empirical Evaluation Narrative Card */}
            <div className="rounded-2xl border border-emerald-500/30 bg-gradient-to-r from-slate-900 via-slate-900/90 to-slate-900 p-6 space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h4 className="text-xs font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4" /> Descriptive Evaluation Narrative: How this Dataset was Analyzed
                </h4>
                <span className={`text-[11px] font-bold px-2.5 py-0.5 rounded-full ${accuracyDelta >= 0 ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'}`}>
                  {higherModel} Leads ({accuracyDelta >= 0 ? `+${accuracyDelta.toFixed(1)}%` : `${accuracyDelta.toFixed(1)}%`})
                </span>
              </div>

              <div className="space-y-2.5 text-xs text-slate-300 leading-relaxed">
                <p>
                  <strong>Empirical Performance on {dsContext.title}:</strong> On this {dsContext.domain.toLowerCase()} dataset, 
                  Quantum KNN attained a test accuracy of <strong className="text-emerald-400">{(results.quantum_knn.accuracy * 100).toFixed(1)}%</strong> and 
                  Macro F1-score of <strong className="text-white">{(results.quantum_knn.f1_macro * 100).toFixed(1)}%</strong> across the held-out test samples, 
                  compared to <strong className="text-indigo-400">{(results.classical_knn.accuracy * 100).toFixed(1)}%</strong> achieved by Scikit-Learn's classical KNN baseline.
                </p>

                <p>
                  <strong>Why Quantum KNN Formed These Boundaries:</strong> By transforming continuous features into rotation angles 
                  <code className="text-cyan-300 bg-slate-950 px-1 py-0.5 rounded border border-slate-800 font-mono">θ = 2·arctan(x)</code> and applying 
                  the <strong className="text-white">{results.feature_map.toUpperCase()}</strong> feature map, the quantum circuit constructs a 
                  multi-qubit entangled state vector in a <strong className="text-cyan-300">{results.qubits_used}-qubit Hilbert space (2^{results.qubits_used} = {2**results.qubits_used} dimensions)</strong>. 
                  Nearest neighbors were identified not by standard Euclidean distance, but via quantum state overlap fidelity 
                  <code className="text-indigo-300 bg-slate-950 px-1 py-0.5 rounded border border-slate-800 font-mono">F = |⟨ψ|φ⟩|²</code>. 
                  {accuracyDelta > 0 
                    ? ` This allowed the quantum model to isolate entangled feature co-dependencies that classical Euclidean geometry struggled to separate.`
                    : accuracyDelta === 0
                    ? ` This indicates that quantum state fidelity overlap preserved the natural topological geometry of the dataset with zero degradation.`
                    : ` For this dataset's localized density distributions, classical Euclidean linear geometry provided sharper margins than bounded trigonometric state overlaps.`}
                </p>

                <p>
                  <strong>Practical Domain Takeaway:</strong> For practical applications in <strong className="text-white">{dsContext.domain}</strong>, 
                  the model provides robust confidence across target classes ({results.classes.join(', ')}). In production on a physical Quantum Processing Unit (QPU), 
                  the state overlap fidelity would be evaluated in constant time via quantum swap tests without classical simulation overhead.
                </p>
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
        );
      })()}

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
      {/* STEP 7: PRE-DOWNLOAD DATASET ANALYSIS & RESEARCH REPORT   */}
      {/* ========================================================= */}
      {currentStep === 7 && datasetId && (
        <div className="space-y-6">
          {/* Header Card */}
          <div className="rounded-2xl border border-cyan-500/30 bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 p-6 shadow-xl">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <span className="inline-flex items-center gap-1.5 rounded-full bg-cyan-500/20 px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-cyan-300 border border-cyan-500/30">
                  <FileText className="h-3.5 w-3.5" /> Executive Research Dossier
                </span>
                <h3 className="text-xl font-black text-white mt-2 flex items-center gap-2">
                  Dataset Analysis Summary & Benchmark Report
                </h3>
                <p className="text-xs text-slate-300 mt-1">
                  Complete technical profile and quantum empirical benchmark results for <strong className="text-cyan-400 font-mono">{overview?.filename || datasetId}</strong>.
                </p>
              </div>

              {/* Direct Download Action Button */}
              <button
                onClick={handleDownloadReport}
                disabled={downloadingReport}
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 px-6 py-3.5 text-xs font-black text-white shadow-xl shadow-cyan-500/25 hover:from-cyan-400 hover:to-indigo-500 transition cursor-pointer flex-shrink-0 disabled:opacity-60"
              >
                {downloadingReport ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    <span>Generating PDF...</span>
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4" />
                    <span>Download Official PDF Report</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Section 1: Ingested Dataset Properties & Statistical Profiling */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 space-y-4">
            <div className="flex items-center gap-2 text-sm font-bold text-white border-b border-slate-800 pb-3">
              <Database className="h-4 w-4 text-cyan-400" />
              <span>1. Dataset Architecture & Quality Health</span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 font-mono">Dataset Name</span>
                <div className="text-sm font-black text-white mt-1 truncate">{overview?.filename || datasetId}</div>
                <div className="text-[11px] text-cyan-400 font-mono mt-0.5">{overview?.memory_usage_mb ?? '--'} MB In-Memory</div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 font-mono">Dimensions</span>
                <div className="text-sm font-black text-white mt-1">{overview?.total_rows ?? '--'} Rows</div>
                <div className="text-[11px] text-slate-400 font-mono mt-0.5">{overview?.total_columns ?? '--'} Total Features</div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 font-mono">Data Quality</span>
                <div className="text-sm font-black text-emerald-400 mt-1">
                  {overview?.missing_value_count === 0 ? '✓ Zero Missing' : `${overview?.missing_value_count} Missing`}
                </div>
                <div className="text-[11px] text-slate-400 font-mono mt-0.5">{overview?.duplicate_record_count ?? 0} Duplicates Handled</div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 font-mono">Target Task</span>
                <div className="text-sm font-black text-cyan-300 mt-1 capitalize">
                  {profile?.task_identification?.subtype || profile?.task_identification?.task_type || 'Supervised Classification'}
                </div>
                <div className="text-[11px] text-indigo-300 font-mono mt-0.5">Target: {configInfo?.target_column || profile?.recommended_target}</div>
              </div>
            </div>

            {/* Target Class Distribution */}
            {profile?.target_distribution && (
              <div className="rounded-xl border border-slate-800/80 bg-slate-950/60 p-4">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 font-mono block mb-2">
                  Target Class Balance & Frequency
                </span>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(profile.target_distribution).map(([cls, cnt]) => (
                    <span key={cls} className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/30 px-3 py-1.5 text-xs text-indigo-300 font-medium">
                      <strong className="text-white capitalize">{cls}:</strong> {cnt} samples ({((cnt / (overview?.total_rows || 1)) * 100).toFixed(1)}%)
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Section 2: Mathematical Preprocessing & Quantum Qubit Mapping */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 space-y-4">
            <div className="flex items-center gap-2 text-sm font-bold text-white border-b border-slate-800 pb-3">
              <Sliders className="h-4 w-4 text-cyan-400" />
              <span>2. Mathematical Preprocessing & Qubit Dimension Mapping</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-1">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 font-mono">Train / Test Split</span>
                <div className="text-sm font-bold text-white">
                  {configInfo?.train_samples ?? '--'} Train / {configInfo?.test_samples ?? '--'} Test
                </div>
                <p className="text-[11px] text-slate-400">Stratified split with zero data leakage (scalers fit only on training data).</p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-1">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 font-mono">Feature Selection & Qubits</span>
                <div className="text-sm font-bold text-cyan-300">
                  {configInfo?.qubit_count ?? '--'} Qubits ({configInfo?.selected_feature_count} features selected)
                </div>
                <div className="text-[11px] text-slate-400 truncate">
                  Features: {configInfo?.selected_feature_names?.join(', ') || 'Auto-selected'}
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-1">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 font-mono">Quantum Rotation Mapping</span>
                <div className="text-sm font-bold text-indigo-300 font-mono">θ = 2 · arctan(x) ∈ [-π, π]</div>
                <p className="text-[11px] text-slate-400">Maps normalized feature values directly to Pauli rotation gate angles.</p>
              </div>
            </div>
          </div>

          {/* Section 3: Empirical Model Performance Comparison */}
          {results && (
            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2 text-sm font-bold text-white">
                  <GitCompare className="h-4 w-4 text-cyan-400" />
                  <span>3. Empirical Benchmark: Classical KNN vs. PennyLane QKNN</span>
                </div>
                <span className="text-xs font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1 rounded-full">
                  Winner: {results.performance_comparison?.higher_accuracy_model || 'Evaluated'}
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-[11px] font-bold uppercase text-slate-400 font-mono">
                      <th className="py-2.5 px-3">Metric</th>
                      <th className="py-2.5 px-3">Classical KNN (Scikit-Learn)</th>
                      <th className="py-2.5 px-3">Quantum KNN (PennyLane)</th>
                      <th className="py-2.5 px-3">Variance (Δ Q - C)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    <tr className="hover:bg-slate-850/50">
                      <td className="py-2.5 px-3 font-sans font-bold text-slate-300">Accuracy</td>
                      <td className="py-2.5 px-3 text-white font-bold">{(results.classical_knn.accuracy * 100).toFixed(2)}%</td>
                      <td className="py-2.5 px-3 text-cyan-300 font-bold">{(results.quantum_knn.accuracy * 100).toFixed(2)}%</td>
                      <td className={`py-2.5 px-3 font-bold ${results.performance_comparison?.accuracy_delta >= 0 ? 'text-emerald-400' : 'text-amber-400'}`}>
                        {results.performance_comparison?.accuracy_delta >= 0 ? '+' : ''}{(results.performance_comparison?.accuracy_delta * 100).toFixed(2)}%
                      </td>
                    </tr>
                    <tr className="hover:bg-slate-850/50">
                      <td className="py-2.5 px-3 font-sans font-bold text-slate-300">Macro F1-Score</td>
                      <td className="py-2.5 px-3 text-white font-bold">{(results.classical_knn.f1_macro * 100).toFixed(2)}%</td>
                      <td className="py-2.5 px-3 text-cyan-300 font-bold">{(results.quantum_knn.f1_macro * 100).toFixed(2)}%</td>
                      <td className={`py-2.5 px-3 font-bold ${results.performance_comparison?.f1_delta >= 0 ? 'text-emerald-400' : 'text-amber-400'}`}>
                        {results.performance_comparison?.f1_delta >= 0 ? '+' : ''}{(results.performance_comparison?.f1_delta * 100).toFixed(2)}%
                      </td>
                    </tr>
                    <tr className="hover:bg-slate-850/50">
                      <td className="py-2.5 px-3 font-sans font-bold text-slate-300">Precision (Macro)</td>
                      <td className="py-2.5 px-3 text-white font-bold">{(results.classical_knn.precision_macro * 100).toFixed(2)}%</td>
                      <td className="py-2.5 px-3 text-cyan-300 font-bold">{(results.quantum_knn.precision_macro * 100).toFixed(2)}%</td>
                      <td className="py-2.5 px-3 text-slate-400">
                        {((results.quantum_knn.precision_macro - results.classical_knn.precision_macro) * 100).toFixed(2)}%
                      </td>
                    </tr>
                    <tr className="hover:bg-slate-850/50">
                      <td className="py-2.5 px-3 font-sans font-bold text-slate-300">Recall (Macro)</td>
                      <td className="py-2.5 px-3 text-white font-bold">{(results.classical_knn.recall_macro * 100).toFixed(2)}%</td>
                      <td className="py-2.5 px-3 text-cyan-300 font-bold">{(results.quantum_knn.recall_macro * 100).toFixed(2)}%</td>
                      <td className="py-2.5 px-3 text-slate-400">
                        {((results.quantum_knn.recall_macro - results.classical_knn.recall_macro) * 100).toFixed(2)}%
                      </td>
                    </tr>
                    <tr className="hover:bg-slate-850/50">
                      <td className="py-2.5 px-3 font-sans font-bold text-slate-300">Prediction Time</td>
                      <td className="py-2.5 px-3 text-slate-400 font-mono">{results.classical_knn.prediction_time_sec}s</td>
                      <td className="py-2.5 px-3 text-amber-300 font-mono">{results.quantum_knn.prediction_time_sec}s</td>
                      <td className="py-2.5 px-3 text-slate-400 font-mono">{results.performance_comparison?.latency_ratio}x ratio</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Quantum Circuit Architecture Telemetry */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                  <span className="text-[10px] font-bold uppercase text-slate-400 block font-mono">Feature Map</span>
                  <span className="text-xs font-black text-cyan-300 capitalize mt-1 block">
                    {results.circuit_summary?.feature_map || 'Hybrid Map'}
                  </span>
                </div>
                <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                  <span className="text-[10px] font-bold uppercase text-slate-400 block font-mono">Total Quantum Gates</span>
                  <span className="text-xs font-black text-white mt-1 block">
                    {results.circuit_summary?.total_gates ?? '--'} Gates
                  </span>
                </div>
                <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                  <span className="text-[10px] font-bold uppercase text-slate-400 block font-mono">CNOT Entanglement</span>
                  <span className="text-xs font-black text-indigo-300 mt-1 block">
                    {results.circuit_summary?.cnot_gates ?? 0} 2-Qubit Entanglers
                  </span>
                </div>
                <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                  <span className="text-[10px] font-bold uppercase text-slate-400 block font-mono">Quantum Simulator</span>
                  <span className="text-xs font-black text-emerald-400 mt-1 block">default.qubit Active</span>
                </div>
              </div>
            </div>
          )}

          {/* Section 4: Academic Takeaways for Guide / Viva Presentation */}
          <div className="rounded-2xl border border-indigo-500/30 bg-gradient-to-r from-slate-900 via-indigo-950/20 to-slate-900 p-6 space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-cyan-300 flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-cyan-400" /> Key Research Findings to Present to Your Project Guide
            </h4>
            <ul className="space-y-2 text-xs text-slate-300 leading-relaxed list-disc list-inside">
              <li>
                <strong>Quantum Feature Encoding:</strong> Tabular features were successfully mapped into multi-qubit superposition and entangled states using parameterized Pauli rotations and CNOT ladder entanglement.
              </li>
              <li>
                <strong>Fidelity-Based Distance Metric:</strong> Nearest neighbors were selected using genuine quantum state overlap fidelity $\mathcal{F} = |\langle\psi|\phi\rangle|^2$, replacing traditional Euclidean distance.
              </li>
              <li>
                <strong>Empirical Performance:</strong> Quantum KNN achieved <strong>{results ? (results.quantum_knn.accuracy * 100).toFixed(1) + '%' : '--'}</strong> accuracy compared to <strong>{results ? (results.classical_knn.accuracy * 100).toFixed(1) + '%' : '--'}</strong> by Classical KNN on an identical stratified test partition.
              </li>
              <li>
                <strong>Hardware Reality Disclosure:</strong> The simulation was executed on PennyLane's <code>default.qubit</code> statevector simulator. Physical QPU deployment on NISQ hardware would compute state overlap directly via quantum swap tests without classical matrix simulation overhead.
              </li>
            </ul>
          </div>

          {/* Bottom Callout & Download Button */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-8 text-center space-y-4">
            <Download className="h-10 w-10 text-cyan-400 mx-auto animate-bounce" />
            <h3 className="text-lg font-black text-white">Ready to Export Official Research Report?</h3>
            <p className="text-xs text-slate-300 max-w-lg mx-auto leading-relaxed">
              Click below to download the complete PDF report generated by ReportLab, incorporating all of the above tables, descriptive distributions, and quantum circuit diagrams for your viva documentation.
            </p>

            <div className="pt-2">
              <button
                onClick={handleDownloadReport}
                disabled={downloadingReport}
                className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 px-8 py-3.5 text-xs font-black text-white shadow-xl shadow-cyan-500/25 hover:from-cyan-400 hover:to-indigo-500 transition cursor-pointer disabled:opacity-60"
              >
                {downloadingReport ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    <span>Generating & Downloading Report...</span>
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4" />
                    <span>Download Official PDF Research Report</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Activity, 
  Play, 
  RefreshCw, 
  Cpu, 
  Settings, 
  FileText, 
  ChevronRight, 
  ChevronDown, 
  ArrowLeft, 
  TrendingUp, 
  BarChart2, 
  Download, 
  XCircle, 
  Trash2,
  MoreVertical,
  Database
} from 'lucide-react';
import './App.css';

const API_BASE = import.meta.env.VITE_API_BASE_URL !== undefined ? import.meta.env.VITE_API_BASE_URL : 'http://localhost:8000';

// ==========================================
// Interactive SVG Chart Component (Lightweight & Offline)
// ==========================================
function MetricsChart({ data, xKey, yKey, yLabel, color, fallbackMsg }) {
  const [hoveredPoint, setHoveredPoint] = useState(null);

  const formatValue = (val) => {
    if (val === null || val === undefined || isNaN(val)) return "N/A";
    if (val === 0) return "0";
    if (Math.abs(val) < 0.0001) {
      return val.toExponential(4);
    }
    if (Math.abs(val) < 0.01) {
      return val.toFixed(6);
    }
    if (Number.isInteger(val)) {
      return val.toString();
    }
    return val.toFixed(4);
  };

  const formatTick = (val) => {
    if (val === 0) return "0";
    if (Math.abs(val) < 0.0001) return val.toExponential(2);
    if (Math.abs(val) < 0.01) return val.toFixed(5);
    if (Math.abs(val) >= 1000) return Math.round(val).toString();
    return val.toFixed(2);
  };

  const validData = (data || []).filter(d => 
    d[xKey] !== undefined && d[xKey] !== null && !isNaN(d[xKey]) &&
    d[yKey] !== undefined && d[yKey] !== null && !isNaN(d[yKey])
  );

  if (validData.length === 0) {
    return (
      <div style={{ height: '180px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', fontSize: '0.85rem', border: '1px dashed rgba(255, 255, 255, 0.08)', borderRadius: '8px', background: 'rgba(15, 23, 42, 0.3)' }}>
        {fallbackMsg || "Waiting for steps to register..."}
      </div>
    );
  }

  const width = 500;
  const height = 180;
  const paddingLeft = 45;
  const paddingRight = 15;
  const paddingTop = 15;
  const paddingBottom = 25;

  if (validData.length === 1) {
    const singlePoint = validData[0];
    const x = paddingLeft + (width - paddingLeft - paddingRight) / 2;
    const y = paddingTop + (height - paddingTop - paddingBottom) / 2;
    return (
      <div className="chart-container" style={{ position: 'relative', flexGrow: 1, minWidth: '280px' }}>
        <div style={{ fontSize: '0.825rem', fontWeight: 600, marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#cbd5e1' }}>{yLabel}</span>
          <span style={{ color: color, fontSize: '0.75rem', fontWeight: 'bold' }}>
            Step: {singlePoint[xKey]} | Value: {formatValue(singlePoint[yKey])}
          </span>
        </div>
        <svg 
          width="100%" 
          height={height} 
          viewBox={`0 0 ${width} ${height}`}
          style={{ overflow: 'visible' }}
        >
          {/* Axis borders */}
          <line x1={paddingLeft} y1={paddingTop} x2={paddingLeft} y2={height - paddingBottom} className="chart-axis-line" />
          <line x1={paddingLeft} y1={height - paddingBottom} x2={width - paddingRight} y2={height - paddingBottom} className="chart-axis-line" />
          
          {/* Draw single dot */}
          <circle cx={x} cy={y} r={6} fill={color} />
          <text x={x} y={y - 12} textAnchor="middle" fill="#cbd5e1" style={{ fontSize: '10px', fontFamily: 'monospace' }}>
            {formatValue(singlePoint[yKey])} (Step {singlePoint[xKey]})
          </text>
        </svg>
      </div>
    );
  }
  const steps = validData.map(d => d[xKey]);
  const vals = validData.map(d => d[yKey]);

  const minX = Math.min(...steps);
  const maxX = Math.max(...steps);
  const rangeX = maxX - minX || 1;

  let minY = Math.min(...vals);
  let maxY = Math.max(...vals);
  if (minY === maxY) {
    minY -= 0.1;
    maxY += 0.1;
  }
  
  if (minY < 0 && (yKey.includes("loss") || yKey.includes("accuracies"))) {
    minY = 0;
  }
  const rangeY = maxY - minY || 1;

  const getX = (x) => paddingLeft + ((x - minX) / rangeX) * (width - paddingLeft - paddingRight);
  const getY = (y) => height - paddingBottom - ((y - minY) / rangeY) * (height - paddingTop - paddingBottom);

  const points = validData.map(d => `${getX(d[xKey])},${getY(d[yKey])}`);
  const dLine = `M ${points.join(' L ')}`;
  const dArea = `${dLine} L ${getX(maxX)},${getY(minY)} L ${getX(minX)},${getY(minY)} Z`;

  const yTicks = 4;
  const yTickVals = Array.from({ length: yTicks }, (_, i) => minY + (i * rangeY) / (yTicks - 1));

  const xTicks = Math.min(5, validData.length);
  const xTickVals = Array.from({ length: xTicks }, (_, i) => minX + (i * rangeX) / (xTicks - 1));

  const handleMouseMove = (e) => {
    const svgRect = e.currentTarget.getBoundingClientRect();
    const clientX = e.clientX - svgRect.left;
    const svgX = (clientX / svgRect.width) * width;
    
    let closest = validData[0];
    let minDiff = Math.abs(getX(validData[0][xKey]) - svgX);

    for (let i = 1; i < validData.length; i++) {
      const diff = Math.abs(getX(validData[i][xKey]) - svgX);
      if (diff < minDiff) {
        minDiff = diff;
        closest = validData[i];
      }
    }
    setHoveredPoint(closest);
  };

  const handleMouseLeave = () => {
    setHoveredPoint(null);
  };

  const gradientId = `grad-${yKey.replace(/[^a-zA-Z0-9]/g, '-')}`;

  return (
    <div className="chart-container" style={{ position: 'relative', flexGrow: 1, minWidth: '280px' }}>
      <div style={{ fontSize: '0.825rem', fontWeight: 600, marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between' }}>
        <span style={{ color: '#cbd5e1' }}>{yLabel}</span>
        {hoveredPoint && (
          <span style={{ color: color, fontSize: '0.75rem', fontWeight: 'bold' }}>
            Step: {hoveredPoint[xKey]} | Value: {formatValue(hoveredPoint[yKey])}
          </span>
        )}
      </div>

      <svg 
        width="100%" 
        height={height} 
        viewBox={`0 0 ${width} ${height}`}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ overflow: 'visible' }}
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.25} />
            <stop offset="100%" stopColor={color} stopOpacity={0.0} />
          </linearGradient>
        </defs>

        {/* Grid lines */}
        {yTickVals.map((val, idx) => {
          const y = getY(val);
          return (
            <g key={idx}>
              <line x1={paddingLeft} y1={y} x2={width - paddingRight} y2={y} className="chart-grid-line" />
              <text x={paddingLeft - 8} y={y + 3} textAnchor="end" className="chart-axis-text">
                {formatTick(val)}
              </text>
            </g>
          );
        })}

        {/* X Axis Labels */}
        {xTickVals.map((val, idx) => {
          const x = getX(val);
          return (
            <text key={idx} x={x} y={height - 5} textAnchor="middle" className="chart-axis-text">
              {Math.round(val)}
            </text>
          );
        })}

        {/* Area fill */}
        <path d={dArea} fill={`url(#${gradientId})`} />

        {/* Line */}
        <path d={dLine} className="chart-line" stroke={color} />

        {/* Axis borders */}
        <line x1={paddingLeft} y1={paddingTop} x2={paddingLeft} y2={height - paddingBottom} className="chart-axis-line" />
        <line x1={paddingLeft} y1={height - paddingBottom} x2={width - paddingRight} y2={height - paddingBottom} className="chart-axis-line" />

        {/* Hover tracker */}
        {hoveredPoint && (
          <g>
            <line 
              x1={getX(hoveredPoint[xKey])} 
              y1={paddingTop} 
              x2={getX(hoveredPoint[xKey])} 
              y2={height - paddingBottom} 
              className="chart-hover-line" 
            />
            <circle 
              cx={getX(hoveredPoint[xKey])} 
              cy={getY(hoveredPoint[yKey])} 
              r={5} 
              fill="#060913" 
              stroke={color} 
              className="chart-dot" 
            />
          </g>
        )}
      </svg>

      {hoveredPoint && (
        <div 
          className="chart-tooltip" 
          style={{ 
            position: 'absolute',
            left: `${Math.min(width - 130, Math.max(paddingLeft, getX(hoveredPoint[xKey]) - 60))}px`,
            top: `${Math.max(paddingTop + 5, getY(hoveredPoint[yKey]) - 65)}px`,
          }}
        >
          <div><strong>Step:</strong> {hoveredPoint[xKey]}</div>
          <div><strong>Val:</strong> {formatValue(hoveredPoint[yKey])}</div>
          {hoveredPoint.epoch !== undefined && <div><strong>Epoch:</strong> {hoveredPoint.epoch}</div>}
        </div>
      )}
    </div>
  );
}

// ==========================================
// Main Cockpit Component
// ==========================================
function App() {
  const [env, setEnv] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(false);
  
  // Single selected job for logs & metrics
  const [selectedJob, setSelectedJob] = useState(null);
  const [logs, setLogs] = useState("");
  const [isPaused, setIsPaused] = useState(false);
  const [pausedStats, setPausedStats] = useState("");
  const [isAdvanced, setIsAdvanced] = useState(false);
  
  // Configuration Settings
  const [showFullLogs, setShowFullLogs] = useState(false);
  const [metricsData, setMetricsData] = useState({ sft: [], dpo: [] });
  const [activeTab, setActiveTab] = useState('cockpit');
  const [metricsPhase, setMetricsPhase] = useState('sft');
  const [wandbConfig, setWandbConfig] = useState({ configured: false, masked_key: null });
  const [apiKeyInput, setApiKeyInput] = useState("");

  // Sync metrics phase with selected job type
  useEffect(() => {
    if (selectedJob) {
      if (selectedJob.type === 'DPO') {
        setMetricsPhase('dpo');
      } else {
        // FULL_PIPELINE or SFT default to sft phase initially
        setMetricsPhase('sft');
      }
    }
  }, [selectedJob]);

  const fetchWandbStatus = async () => {
    try {
      const res = await axios.get(`${API_BASE}/settings/wandb`);
      setWandbConfig(res.data);
    } catch (err) {
      console.error("Failed to fetch WandB status:", err);
    }
  };

  useEffect(() => {
    fetchWandbStatus();
  }, [activeTab]);

  const [config, setConfig] = useState({
    model_id: "Qwen/Qwen2.5-0.5B-Instruct",
    dataset_path: "test_datasets/standard_test_data.csv",
    dataset_len: 1000,
    batch_size: 1,
    grad_accum: 2,
    epochs: 1,
    learning_rate: 0.000005,
    beta: 0.3,
    use_4bit: true,
    lora_r: 8,
    lora_alpha: 16,
    lora_dropout: 0.05,
    max_prompt_length: 256,
    max_seq_length: 384,
    min_prompt_len: 10,
    min_response_len: 3,
    max_response_len: 300,
    max_length_diff: 100,
    wandb_mode: "disabled",
    wandb_project: "rlhf-handson-pytorch"
  });

  // Poll environment status and recent jobs list
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [envRes, jobsRes] = await Promise.all([
          axios.get(`${API_BASE}/env-check`),
          axios.get(`${API_BASE}/jobs`)
        ]);
        setEnv(envRes.data);
        setJobs(jobsRes.data);

        if (selectedJob) {
          const updated = jobsRes.data.find(j => j.id === selectedJob.id);
          if (updated && (updated.status !== selectedJob.status || updated.log_path !== selectedJob.log_path)) {
            setSelectedJob(updated);
          }
        } else if (jobsRes.data.length > 0) {
          const sortedJobs = [...jobsRes.data].sort((a, b) => b.id - a.id);
          setSelectedJob(sortedJobs[0]);
        }
      } catch (err) {
        console.error("Failed to fetch dashboard data:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [selectedJob]);

  // Poll logs for the selected job
  useEffect(() => {
    if (!selectedJob) return;
    
    const fetchLogs = async () => {
      try {
        const limit = showFullLogs ? -1 : 150;
        const res = await axios.get(`${API_BASE}/jobs/${selectedJob.id}/logs?limit=${limit}`);
        setLogs(res.data.logs);
        setIsPaused(res.data.paused);
        setPausedStats(res.data.paused_stats);
      } catch (err) {
        setLogs("Waiting for logs to initialize...");
      }
    };

    fetchLogs();
    const interval = setInterval(fetchLogs, 3000);
    return () => clearInterval(interval);
  }, [selectedJob, showFullLogs]);

  // Poll metrics for the selected job
  useEffect(() => {
    if (!selectedJob) return;
    
    const fetchMetrics = async () => {
      try {
        const res = await axios.get(`${API_BASE}/jobs/${selectedJob.id}/metrics`);
        setMetricsData(res.data);
      } catch (err) {
        console.error("Failed to fetch metrics:", err);
      }
    };

    fetchMetrics();
    
    let interval;
    if (selectedJob.status === 'RUNNING' || selectedJob.status === 'AWAITING_APPROVAL' || selectedJob.status === 'PENDING') {
      interval = setInterval(fetchMetrics, 4000);
    }
    return () => clearInterval(interval);
  }, [selectedJob]);

  // Launch training job
  const startTraining = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/jobs/train`, config);
      setSelectedJob(res.data);
      setLogs("Launching background training process...");
      setShowFullLogs(false);
      setMetricsData({ sft: [], dpo: [] });
    } catch (err) {
      alert("Failed to start training: " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const approveJob = async () => {
    try {
      await axios.post(`${API_BASE}/jobs/${selectedJob.id}/approve`);
      setIsPaused(false);
    } catch (err) {
      alert("Failed to approve job");
    }
  };

  const abortJob = async (jobId) => {
    try {
      await axios.post(`${API_BASE}/jobs/${jobId}/abort`);
      setIsPaused(false);
      if (selectedJob && selectedJob.id === jobId) {
        setSelectedJob(prev => ({ ...prev, status: 'ABORTED' }));
      }
    } catch (err) {
      alert("Failed to abort job");
    }
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setConfig(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : (type === 'number' ? parseFloat(value) : value)
    }));
  };

  return (
    <div className="container">
      {/* HEADER SECTION */}
      <header className="header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <TrendingUp size={28} color="#60a5fa" />
          <h1>RLHF Dashboard Cockpit</h1>
        </div>

        {/* View Toggle Tabs */}
        <div style={{ display: 'flex', gap: '0.35rem', background: 'rgba(15, 23, 42, 0.4)', padding: '0.25rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
          <button 
            onClick={() => setActiveTab('cockpit')}
            style={{ 
              background: activeTab === 'cockpit' ? '#2563eb' : 'transparent',
              color: activeTab === 'cockpit' ? '#ffffff' : '#94a3b8',
              padding: '0.45rem 1.1rem',
              fontSize: '0.85rem',
              borderRadius: '6px',
              border: 'none',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            <Cpu size={15} />
            Training Cockpit
          </button>
          <button 
            onClick={() => setActiveTab('metrics')}
            style={{ 
              background: activeTab === 'metrics' ? '#2563eb' : 'transparent',
              color: activeTab === 'metrics' ? '#ffffff' : '#94a3b8',
              padding: '0.45rem 1.1rem',
              fontSize: '0.85rem',
              borderRadius: '6px',
              border: 'none',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            <BarChart2 size={15} />
            Analysis Charts
          </button>
          <button 
            onClick={() => setActiveTab('history')}
            style={{ 
              background: activeTab === 'history' ? '#2563eb' : 'transparent',
              color: activeTab === 'history' ? '#ffffff' : '#94a3b8',
              padding: '0.45rem 1.1rem',
              fontSize: '0.85rem',
              borderRadius: '6px',
              border: 'none',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            <Activity size={15} />
            Recent Job History
            {jobs.length > 0 && (
              <span style={{ 
                background: activeTab === 'history' ? 'rgba(255,255,255,0.25)' : 'rgba(59,130,246,0.15)', 
                color: activeTab === 'history' ? '#ffffff' : '#60a5fa', 
                fontSize: '0.725rem', 
                padding: '0.1rem 0.45rem', 
                borderRadius: '9999px',
                marginLeft: '0.25rem',
                fontWeight: 700
              }}>
                {jobs.length}
              </span>
            )}
          </button>
          <button 
            onClick={() => setActiveTab('wandb')}
            style={{ 
              background: activeTab === 'wandb' ? '#2563eb' : 'transparent',
              color: activeTab === 'wandb' ? '#ffffff' : '#94a3b8',
              padding: '0.45rem 1.1rem',
              fontSize: '0.85rem',
              borderRadius: '6px',
              border: 'none',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            <Database size={15} />
            WandB Integration
          </button>
        </div>

        {env && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div className={`status-badge status-${env.status}`}>
              {env.status === 'ready' ? 'System Ready' : env.status}
            </div>
          </div>
        )}
      </header>

      {/* activeTab === 'cockpit' View */}
      {activeTab === 'cockpit' && (
        <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 4fr) minmax(0, 6fr)', alignItems: 'start' }}>
        
        {/* LEFT COLUMN: Environment Details & Training Configuration Form */}
        <div style={{ minWidth: 0 }}>
          {/* Environment card */}
          <section className="card" style={{ minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <Cpu size={20} color="#60a5fa" />
              <h2 style={{ margin: 0, fontSize: '1.25rem' }}>Environment</h2>
            </div>
            {env ? (
              <div style={{ fontSize: '0.875rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.50rem' }}>
                <div><span style={{ color: '#94a3b8' }}>GPU:</span> <strong style={{ color: '#e2e8f0' }}>{env.devices[0]?.name || "None (CPU)"}</strong></div>
                <div><span style={{ color: '#94a3b8' }}>VRAM:</span> <strong style={{ color: '#e2e8f0' }}>{env.devices[0]?.vram || "N/A"}</strong></div>
                <div><span style={{ color: '#94a3b8' }}>CUDA Support:</span> <strong style={env.cuda_available ? { color: '#34d399' } : { color: '#f87171' }}>{env.cuda_available ? "Yes" : "No"}</strong></div>
                <div><span style={{ color: '#94a3b8' }}>BitsAndBytes:</span> <strong style={env.bitsandbytes_available ? { color: '#34d399' } : { color: '#f87171' }}>{env.bitsandbytes_available ? "Loaded" : "Failed"}</strong></div>
              </div>
            ) : (
              <div style={{ color: '#94a3b8', fontSize: '0.875rem' }}>Connecting to training backend...</div>
            )}
          </section>

          {/* SFT / DPO Parameter Knob Form */}
          <section className="card" style={{ minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
              <Settings size={20} color="#94a3b8" />
              <h2 style={{ margin: 0, fontSize: '1.25rem' }}>Training Configuration</h2>
            </div>
            
            <div className="form-group" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <label>Model ID / Path (HuggingFace or Local)</label>
                <input className="input" name="model_id" value={config.model_id} onChange={handleInputChange} />
              </div>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                <div>
                  <label>Local Dataset Path</label>
                  <input className="input" name="dataset_path" value={config.dataset_path || ""} onChange={handleInputChange} placeholder="e.g. data/my_data.csv" />
                </div>
                <div>
                  <label>Max Training Rows</label>
                  <input className="input" type="number" name="dataset_len" value={config.dataset_len} onChange={handleInputChange} />
                </div>
              </div>

              <div style={{ background: 'rgba(15, 23, 42, 0.4)', padding: '0.85rem', borderRadius: '8px', border: '1px dashed rgba(255,255,255,0.06)' }}>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8', lineHeight: '1.4' }}>
                  <strong style={{ color: '#60a5fa' }}>Strict Schema Requirement:</strong> Ingestion file must contain exactly three column headers: <code>prompt</code>, <code>chosen</code>, and <code>rejected</code>.
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem', alignItems: 'center' }}>
                <div>
                  <label>Batch Size</label>
                  <input className="input" type="number" name="batch_size" value={config.batch_size} onChange={handleInputChange} />
                </div>
                <div>
                  <label>Epochs</label>
                  <input className="input" type="number" step="0.1" name="epochs" value={config.epochs} onChange={handleInputChange} />
                </div>
                <div style={{ paddingTop: '1.3rem' }}>
                  <label className="checkbox-container">
                    <input type="checkbox" name="use_4bit" checked={config.use_4bit} onChange={handleInputChange} />
                    <span>Use 4-bit</span>
                  </label>
                </div>
              </div>

              {/* Advanced Settings Toggle */}
              <div style={{ marginTop: '0.25rem' }}>
                <button 
                  type="button"
                  onClick={() => setIsAdvanced(!isAdvanced)} 
                  className="btn-secondary"
                  style={{ padding: '0.35rem 0.65rem', fontSize: '0.8rem', width: 'auto' }}
                >
                  {isAdvanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  {isAdvanced ? 'Hide Advanced Settings' : 'Show Advanced Settings'}
                </button>
              </div>

              {isAdvanced && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem', marginTop: '0.25rem' }}>
                  
                  {/* Optimization */}
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.5rem', letterSpacing: '0.05em', textTransform: 'uppercase' }}>Optimization Parameters</div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '0.75rem' }}>
                      <div>
                        <label>Learning Rate</label>
                        <input className="input" type="number" step="0.000001" name="learning_rate" value={config.learning_rate} onChange={handleInputChange} />
                      </div>
                      <div>
                        <label>Grad Accumulation</label>
                        <input className="input" type="number" name="grad_accum" value={config.grad_accum} onChange={handleInputChange} />
                      </div>
                    </div>
                  </div>

                  {/* LoRA */}
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.5rem', letterSpacing: '0.05em', textTransform: 'uppercase' }}>LoRA Architecture</div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1.2fr', gap: '0.75rem' }}>
                      <div>
                        <label>Rank (R)</label>
                        <input className="input" type="number" name="lora_r" value={config.lora_r} onChange={handleInputChange} />
                      </div>
                      <div>
                        <label>Alpha</label>
                        <input className="input" type="number" name="lora_alpha" value={config.lora_alpha} onChange={handleInputChange} />
                      </div>
                      <div>
                        <label>Dropout</label>
                        <input className="input" type="number" step="0.01" name="lora_dropout" value={config.lora_dropout} onChange={handleInputChange} />
                      </div>
                    </div>
                  </div>

                  {/* Data Filtering */}
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.5rem', letterSpacing: '0.05em', textTransform: 'uppercase' }}>Data & Formatting Filters</div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                      <div>
                        <label>Max Seq Length</label>
                        <input className="input" type="number" name="max_seq_length" value={config.max_seq_length} onChange={handleInputChange} />
                      </div>
                      <div>
                        <label>Max Response Length</label>
                        <input className="input" type="number" name="max_response_len" value={config.max_response_len} onChange={handleInputChange} />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <button onClick={startTraining} disabled={loading} style={{ marginTop: '0.75rem', width: '100%', padding: '0.85rem' }}>
                {loading ? 'Launching training...' : 'Run Full RLHF Pipeline'}
              </button>
            </div>
          </section>
        </div>

        {/* RIGHT COLUMN: Active Metrics Charts (Top) & Console Logs (Bottom) */}
        <div style={{ minWidth: 0 }}>
          
          {/* Active Run Console Logs Card */}
          {selectedJob && (
            <section className="card" style={{ borderLeft: '4px solid var(--running)', overflow: 'hidden', maxWidth: '100%', minWidth: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <FileText size={18} color="#60a5fa" />
                  <h2 style={{ margin: 0, fontSize: '1.1rem' }}>Console Logs</h2>
                </div>
                {isPaused && (
                  <div className="status-badge status-warning" style={{ fontSize: '0.65rem' }}>Awaiting Approval</div>
                )}
              </div>

              {isPaused && (
                <div style={{ background: 'rgba(30, 27, 75, 0.6)', border: '1px solid #4338ca', padding: '1rem', borderRadius: '8px', marginBottom: '1rem' }}>
                  <div style={{ fontWeight: 600, color: '#c7d2fe', marginBottom: '0.25rem', fontSize: '0.875rem' }}>⚠️ Quality Gate: Significant Data Loss</div>
                  <div style={{ fontSize: '0.775rem', color: '#94a3b8', marginBottom: '0.75rem' }}>
                    Keep ratio is low: <strong>{pausedStats}</strong> of data matched constraints. Do you want to proceed?
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button onClick={approveJob} style={{ background: '#059669', fontSize: '0.75rem', padding: '0.35rem 0.85rem' }}>Proceed anyway</button>
                    <button onClick={() => abortJob(selectedJob.id)} className="btn-danger" style={{ fontSize: '0.75rem', padding: '0.35rem 0.85rem' }}>Abort Job</button>
                  </div>
                </div>
              )}

              {/* Log pre block with strict sizing rules */}
              <pre 
                className="console-log" 
                style={{ 
                  height: '280px', 
                  margin: 0, 
                  overflow: 'auto', 
                  whiteSpace: 'pre-wrap', 
                  wordBreak: 'break-all', 
                  maxWidth: '100%',
                  boxSizing: 'border-box'
                }}
              >
                {logs || "Waiting for pipeline runner output..."}
              </pre>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.75rem' }}>
                <label className="checkbox-container" style={{ fontSize: '0.75rem' }}>
                  <input type="checkbox" checked={showFullLogs} onChange={(e) => setShowFullLogs(e.target.checked)} />
                  <span>Load Full Log File (bypass 150 lines tail)</span>
                </label>
              </div>
            </section>
          )}

          {/* Active Run Performance Metrics Card */}
          <section className="card" style={{ overflow: 'hidden', maxWidth: '100%', minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <BarChart2 size={18} color="#60a5fa" />
                <h2 style={{ margin: 0, fontSize: '1.1rem' }}>
                  Performance Metrics {selectedJob ? `#${selectedJob.id}` : ''}
                </h2>
              </div>
              {selectedJob && (
                <span className={`status-badge status-${selectedJob.status.toLowerCase()}`}>
                  {selectedJob.status}
                </span>
              )}
            </div>

            {selectedJob ? (
              selectedJob.type === 'DUMMY' ? (
                <div style={{ color: '#94a3b8', fontSize: '0.875rem', height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  Metrics not applicable for dummy tasks.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  {/* SFT Charts */}
                  {(selectedJob.type === 'FULL_PIPELINE' || selectedJob.type === 'SFT') && (
                    <div>
                      <MetricsChart 
                        data={metricsData.sft} 
                        xKey="step" 
                        yKey="loss" 
                        yLabel="Phase 1: SFT Cross-Entropy Loss" 
                        color="#c084fc" 
                      />
                    </div>
                  )}
                  {/* DPO Charts (Responsive horizontal fit) */}
                  {(selectedJob.type === 'FULL_PIPELINE' || selectedJob.type === 'DPO') && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
                      <MetricsChart 
                        data={metricsData.dpo} 
                        xKey="step" 
                        yKey="loss" 
                        yLabel="Phase 2: DPO Training Loss" 
                        color="#2dd4bf" 
                      />
                      <MetricsChart 
                        data={metricsData.dpo} 
                        xKey="step" 
                        yKey="rewards/accuracies" 
                        yLabel="Phase 2: DPO Choice Accuracy" 
                        color="#fb923c" 
                      />
                    </div>
                  )}
                </div>
              )
            ) : (
              <div style={{ color: '#94a3b8', fontSize: '0.875rem', height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                No active run selected. Select a run from the history list to view charts.
              </div>
            )}
          </section>
        </div>
      </div>
      )}

      {/* activeTab === 'metrics' View */}
      {activeTab === 'metrics' && (
        <div style={{ width: '100%' }}>
          {/* Top Selection/Control Header */}
          <section className="card" style={{ minWidth: 0, padding: '1.5rem', marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <BarChart2 size={24} color="#60a5fa" />
                <div>
                  <h2 style={{ margin: 0, fontSize: '1.3rem' }}>Analysis & Deep Insights</h2>
                  <p style={{ margin: 0, fontSize: '0.8rem', color: '#94a3b8' }}>
                    Monitor detailed learning curves, evaluation metrics, and GPU telemetry in real-time.
                  </p>
                </div>
              </div>

              {/* Job Selector Dropdown */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Selected Job:</span>
                <select
                  value={selectedJob ? selectedJob.id : ''}
                  onChange={(e) => {
                    const job = jobs.find(j => j.id === parseInt(e.target.value));
                    if (job) setSelectedJob(job);
                  }}
                  className="input"
                  style={{ width: '220px', padding: '0.45rem 0.75rem', background: '#0f172a', borderRadius: '6px' }}
                >
                  <option value="" disabled>-- Select a Job --</option>
                  {[...jobs].sort((a, b) => b.id - a.id).map(job => (
                    <option key={job.id} value={job.id}>
                      Job #{job.id} ({job.type}) - {job.status}
                    </option>
                  ))}
                </select>

                {selectedJob && (
                  <span className={`status-badge status-${selectedJob.status.toLowerCase()}`}>
                    {selectedJob.status}
                  </span>
                )}
              </div>
            </div>

            {/* SFT vs DPO Phase Toggle (Only show if job is FULL_PIPELINE) */}
            {selectedJob && selectedJob.type === 'FULL_PIPELINE' && (
              <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1.25rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
                <button
                  onClick={() => setMetricsPhase('sft')}
                  className={metricsPhase === 'sft' ? '' : 'btn-secondary'}
                  style={{ padding: '0.4rem 1rem', fontSize: '0.8rem' }}
                >
                  Phase 1: SFT Metrics
                </button>
                <button
                  onClick={() => setMetricsPhase('dpo')}
                  className={metricsPhase === 'dpo' ? '' : 'btn-secondary'}
                  style={{ padding: '0.4rem 1rem', fontSize: '0.8rem' }}
                >
                  Phase 2: DPO Metrics
                </button>
              </div>
            )}
          </section>

          {/* Validation of Selected Job */}
          {!selectedJob ? (
            <section className="card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
              <BarChart2 size={48} color="#475569" style={{ marginBottom: '1rem', marginLeft: 'auto', marginRight: 'auto' }} />
              <div style={{ color: '#94a3b8', fontSize: '0.95rem' }}>
                No active run selected. Select a run using the dropdown above or from the history list.
              </div>
            </section>
          ) : selectedJob.type === 'DUMMY' ? (
            <section className="card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
              <BarChart2 size={48} color="#475569" style={{ marginBottom: '1rem', marginLeft: 'auto', marginRight: 'auto' }} />
              <div style={{ color: '#94a3b8', fontSize: '0.95rem' }}>
                Detailed analysis metrics are not applicable for DUMMY tasks.
              </div>
            </section>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              
              {/* 1. TRAINING & MODEL CURVES */}
              <section className="card" style={{ margin: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
                  <TrendingUp size={20} color="#c084fc" />
                  <h3 style={{ margin: 0, fontSize: '1.15rem', color: '#e2e8f0' }}>
                    Model Training Curves ({metricsPhase === 'sft' ? 'Phase 1: SFT' : 'Phase 2: DPO'})
                  </h3>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: '1.5rem' }}>
                  <MetricsChart 
                    data={metricsData[metricsPhase]} 
                    xKey="step" 
                    yKey="loss" 
                    yLabel="Training Loss vs. Global Step" 
                    color="#c084fc" 
                    fallbackMsg="Waiting for training loss logs..."
                  />
                  <MetricsChart 
                    data={metricsData[metricsPhase]} 
                    xKey="step" 
                    yKey="eval_loss" 
                    yLabel="Evaluation Loss vs. Step" 
                    color="#f59e0b" 
                    fallbackMsg="No evaluation step has run yet (needs SFT/DPO eval_steps interval)."
                  />
                  <MetricsChart 
                    data={metricsData[metricsPhase]} 
                    xKey="step" 
                    yKey="learning_rate" 
                    yLabel="Learning Rate vs. Step" 
                    color="#3b82f6" 
                    fallbackMsg="Learning rate data not logged yet."
                  />
                  <MetricsChart 
                    data={metricsData[metricsPhase]} 
                    xKey="step" 
                    yKey="grad_norm" 
                    yLabel="Gradient Norm vs. Step" 
                    color="#2dd4bf" 
                    fallbackMsg="Gradient norm data not logged yet."
                  />
                </div>
              </section>

              {/* 2. GPU TELEMETRY & HARDWARE RESOURCE INSIGHTS */}
              <section className="card" style={{ margin: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
                  <Cpu size={20} color="#34d399" />
                  <h3 style={{ margin: 0, fontSize: '1.15rem', color: '#e2e8f0' }}>
                    GPU Telemetry & Hardware Performance ({metricsPhase === 'sft' ? 'SFT Phase' : 'DPO Phase'})
                  </h3>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: '1.5rem' }}>
                  {/* GPU Memory: Allocated vs Reserved (PyTorch Caching) */}
                  <MetricsChart 
                    data={metricsData[metricsPhase]} 
                    xKey="step" 
                    yKey="gpu/allocated_mb" 
                    yLabel="Cuda Memory Allocated (MB)" 
                    color="#10b981" 
                    fallbackMsg="Cuda memory metrics not found (run might be on CPU)."
                  />
                  <MetricsChart 
                    data={metricsData[metricsPhase]} 
                    xKey="step" 
                    yKey="gpu/reserved_mb" 
                    yLabel="Cuda Memory Reserved (MB)" 
                    color="#3b82f6" 
                    fallbackMsg="Cuda reserved memory metrics not found."
                  />

                  {/* GPU Clocks, Temperature, Power usage */}
                  <MetricsChart 
                    data={metricsData[metricsPhase]} 
                    xKey="step" 
                    yKey="gpu/power_w" 
                    yLabel="GPU Power Draw (Watts)" 
                    color="#f43f5e" 
                    fallbackMsg="NVML power consumption data not available."
                  />
                  <MetricsChart 
                    data={metricsData[metricsPhase]} 
                    xKey="step" 
                    yKey="gpu/clock_mhz" 
                    yLabel="Graphics Clock Speed (MHz)" 
                    color="#fb923c" 
                    fallbackMsg="NVML clock speed data not available."
                  />
                  <MetricsChart 
                    data={metricsData[metricsPhase]} 
                    xKey="step" 
                    yKey="gpu/temp_c" 
                    yLabel="GPU Temperature (°C)" 
                    color="#f59e0b" 
                    fallbackMsg="NVML temperature data not available."
                  />
                  <MetricsChart 
                    data={metricsData[metricsPhase]} 
                    xKey="step" 
                    yKey="gpu/used_mb" 
                    yLabel="System GPU Memory Used (NVML - MB)" 
                    color="#6366f1" 
                    fallbackMsg="NVML hardware memory utilization data not available."
                  />
                </div>
              </section>
              
            </div>
          )}
        </div>
      )}

      {/* activeTab === 'history' View */}
      {activeTab === 'history' && (
        <div style={{ width: '100%' }}>
          <section className="card" style={{ minWidth: 0, padding: '2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Activity size={22} color="#f59e0b" />
                <h2 style={{ margin: 0, fontSize: '1.4rem' }}>Job Execution Archive</h2>
              </div>
              {jobs.length > 0 && (
                <button 
                  onClick={async () => {
                    if(confirm("Permanently wipe all logs, artifacts, and training outputs for every historical run?")) {
                      try {
                        await axios.delete(`${API_BASE}/jobs`);
                        setSelectedJob(null);
                      } catch (err) { alert("Failed to clear database"); }
                    }
                  }}
                  className="btn-danger"
                  style={{ padding: '0.5rem 1.2rem', fontSize: '0.85rem' }}
                >
                  <Trash2 size={16} />
                  Wipe Database
                </button>
              )}
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Parameters</th>
                    <th>Actions & Toggles</th>
                    <th>Management</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.length === 0 ? (
                    <tr>
                      <td colSpan="6" style={{ textAlign: 'center', color: '#94a3b8', padding: '3rem', fontSize: '0.95rem' }}>
                        No runs executed yet. Start a run from the Cockpit.
                      </td>
                    </tr>
                  ) : (
                    [...jobs].reverse().map(job => (
                      <tr key={job.id} style={selectedJob?.id === job.id ? { background: 'rgba(59, 130, 246, 0.05)' } : {}}>
                        <td><strong>#{job.id}</strong></td>
                        <td><span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8' }}>{job.type}</span></td>
                        <td>
                          <span className={`status-badge status-${job.status.toLowerCase()}`}>
                            {job.status}
                          </span>
                        </td>
                        <td>
                          <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                            Model: {job.config?.model_id?.split('/').pop() || 'N/A'} | Epochs: {job.config?.epochs}
                          </div>
                        </td>
                        <td>
                          <div style={{ display: 'flex', gap: '0.5rem' }}>
                            <button 
                              onClick={() => { 
                                setSelectedJob(job); 
                                setShowFullLogs(false); 
                                setActiveTab('cockpit'); 
                              }} 
                              className="btn-secondary" 
                              style={{ 
                                padding: '0.35rem 0.85rem', 
                                fontSize: '0.75rem',
                                background: selectedJob?.id === job.id ? 'rgba(59, 130, 246, 0.1)' : 'rgba(75, 85, 99, 0.3)',
                                borderColor: selectedJob?.id === job.id ? 'rgba(59, 130, 246, 0.2)' : 'rgba(255, 255, 255, 0.08)'
                              }}
                            >
                              Inspect Performance & Logs
                            </button>
                            {job.status === 'COMPLETED' && (
                              <a 
                                href={`${API_BASE}/jobs/${job.id}/download`} 
                                download
                                style={{
                                  padding: '0.35rem 0.85rem', 
                                  fontSize: '0.75rem', 
                                  background: 'rgba(16, 185, 129, 0.15)', 
                                  color: '#34d399', 
                                  border: '1px solid rgba(16, 185, 129, 0.2)',
                                  borderRadius: '8px',
                                  textDecoration: 'none', 
                                  display: 'inline-flex',
                                  alignItems: 'center',
                                  gap: '0.25rem'
                                }}
                              >
                                <Download size={12} /> Zip
                              </a>
                            )}
                          </div>
                        </td>
                        <td>
                          <div style={{ display: 'flex', gap: '0.5rem' }}>
                            {(job.status === 'RUNNING' || job.status === 'AWAITING_APPROVAL') && (
                              <button 
                                onClick={() => {
                                  if(confirm(`Abort training run #${job.id}?`)) {
                                    abortJob(job.id);
                                  }
                                }} 
                                className="btn-danger"
                                style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
                              >
                                Kill
                              </button>
                            )}
                            {!(job.status === 'RUNNING' || job.status === 'AWAITING_APPROVAL') && (
                              <button 
                                onClick={async () => {
                                  if(confirm(`Wipe run #${job.id} files and remove from database?`)) {
                                      try {
                                        await axios.delete(`${API_BASE}/jobs/${job.id}`);
                                        if (selectedJob && selectedJob.id === job.id) setSelectedJob(null);
                                      } catch (err) { alert(err.response?.data?.detail || "Delete failed"); }
                                  }
                                }} 
                                className="btn-secondary"
                                style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
                              >
                                Wipe
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      )}

      {/* activeTab === 'wandb' View */}
      {activeTab === 'wandb' && (
        <div style={{ width: '100%' }}>
          <section className="card" style={{ minWidth: 0, padding: '2.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '1rem' }}>
              <Database size={24} color="#3b82f6" />
              <h2 style={{ margin: 0, fontSize: '1.4rem' }}>Weights & Biases (WandB) Connection</h2>
            </div>

            <p style={{ color: '#94a3b8', fontSize: '0.9rem', lineHeight: '1.6', marginBottom: '2rem', maxWidth: '800px' }}>
              Link your local machine to Weights & Biases (WandB) to aggregate training performance, loss statistics, and choice validation curves.
              This key is securely stored in your local database and will be used automatically to synchronize telemetries for all subsequent training jobs.
            </p>

            <div style={{ background: 'rgba(15, 23, 42, 0.4)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '2rem', maxWidth: '600px' }}>
              <h3 style={{ margin: '0 0 1.25rem 0', fontSize: '1.1rem', color: '#f8fafc' }}>Connection Status</h3>
              
              {wandbConfig.configured ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <span className="status-badge status-ready" style={{ fontSize: '0.75rem', padding: '0.4rem 0.85rem' }}>Connected</span>
                    <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>API key verified & locked.</span>
                  </div>
                  
                  <div style={{ background: 'rgba(9, 13, 22, 0.6)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '8px', padding: '0.85rem 1rem' }}>
                    <div style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>Stored Credential Profile</div>
                    <code style={{ fontSize: '0.85rem', color: '#60a5fa', fontFamily: 'monospace' }}>{wandbConfig.masked_key}</code>
                  </div>

                  <div style={{ marginTop: '0.5rem' }}>
                    <button 
                      onClick={async () => {
                        if (confirm("Disconnect and permanently delete your Weights & Biases API Key from database settings?")) {
                          try {
                            await axios.delete(`${API_BASE}/settings/wandb`);
                            alert("WandB API key deleted successfully.");
                            fetchWandbStatus();
                          } catch (err) {
                            alert("Failed to disconnect WandB: " + (err.response?.data?.detail || err.message));
                          }
                        }
                      }}
                      className="btn-danger"
                      style={{ padding: '0.65rem 1.25rem', fontSize: '0.85rem', width: 'auto' }}
                    >
                      Disconnect & Wipe Key
                    </button>
                  </div>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <span className="status-badge status-warning" style={{ fontSize: '0.75rem', padding: '0.4rem 0.85rem' }}>Offline / Not Configured</span>
                  </div>
                  
                  <p style={{ margin: 0, fontSize: '0.825rem', color: '#94a3b8', lineHeight: '1.4' }}>
                    You have not configured an API Key. Training telemetry will only be logged locally to the metrics charts. To enable WandB tracking, paste your API key below:
                  </p>

                  <div className="form-group" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    <label style={{ fontSize: '0.75rem', color: '#94a3b8' }}>WandB API Key</label>
                    <input 
                      type="password" 
                      placeholder="Paste your W&B API Key (typically 40 characters)" 
                      value={apiKeyInput}
                      onChange={(e) => setApiKeyInput(e.target.value)}
                      className="input"
                      style={{ maxWidth: '100%', fontFamily: 'monospace' }}
                    />
                    <div style={{ fontSize: '0.725rem', color: '#64748b', marginTop: '0.25rem' }}>
                      Don't have a key? Retrieve it from your official <a href="https://wandb.ai/authorize" target="_blank" rel="noopener noreferrer" style={{ color: '#3b82f6', textDecoration: 'underline' }}>WandB Authorize Portal</a>.
                    </div>
                  </div>

                  <div style={{ marginTop: '0.5rem' }}>
                    <button 
                      onClick={async () => {
                        const key = apiKeyInput.trim();
                        if (!key) {
                          alert("Please paste a valid WandB API Key first.");
                          return;
                        }
                        try {
                          await axios.post(`${API_BASE}/settings/wandb`, { api_key: key });
                          alert("API key successfully connected and stored!");
                          setApiKeyInput("");
                          fetchWandbStatus();
                        } catch (err) {
                          alert("Failed to save WandB API key: " + (err.response?.data?.detail || err.message));
                        }
                      }}
                      style={{ padding: '0.65rem 1.25rem', fontSize: '0.85rem', width: 'auto', background: '#2563eb' }}
                    >
                      Save & Establish Connection
                    </button>
                  </div>
                </div>
              )}
            </div>
          </section>
        </div>
      )}

      {/* FOOTER */}
      <footer style={{ marginTop: '3rem', borderTop: '1px solid var(--border-color)', paddingTop: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', color: '#64748b' }}>
        <div>Local-First RLHF framework | Enterprise GDPR Compliant</div>
        <div>Offline Cockpit v1.2.0</div>
      </footer>

      {/* STYLES FOR DYNAMIC SPINNING */}
      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        .animate-spin {
          animation: spin 1s linear infinite;
        }
      `}} />
    </div>
  );
}

export default App;

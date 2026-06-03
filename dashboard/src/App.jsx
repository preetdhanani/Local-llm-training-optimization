import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, Play, RefreshCw, Cpu, Settings, FileText, ChevronRight, ChevronDown } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL !== undefined ? import.meta.env.VITE_API_BASE_URL : 'http://localhost:8000';

function App() {
  const [env, setEnv] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedJob, setSelectedJob] = useState(null);
  const [logs, setLogs] = useState("");
  const [isPaused, setIsPaused] = useState(false);
  const [pausedStats, setPausedStats] = useState("");
  const [isAdvanced, setIsAdvanced] = useState(false);
  
  // Training Form State
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

  // Poll for data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [envRes, jobsRes] = await Promise.all([
          axios.get(`${API_BASE}/env-check`),
          axios.get(`${API_BASE}/jobs`)
        ]);
        setEnv(envRes.data);
        setJobs(jobsRes.data);
      } catch (err) {
        console.error("Failed to fetch data:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  // Poll for logs if a job is selected
  useEffect(() => {
    if (!selectedJob) return;
    
    const fetchLogs = async () => {
      try {
        const res = await axios.get(`${API_BASE}/jobs/${selectedJob.id}/logs`);
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
  }, [selectedJob]);

  const startTraining = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/jobs/train`, config);
      setSelectedJob(res.data);
      setLogs("Initializing training process...");
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

  const abortJob = async () => {
    try {
      await axios.post(`${API_BASE}/jobs/${selectedJob.id}/abort`);
      setIsPaused(false);
      setSelectedJob(null);
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
      <header className="header">
        <h1>RLHF Dashboard</h1>
        {env && (
          <div className={`status-badge status-${env.status}`}>
            {env.status === 'ready' ? 'System Ready' : env.status}
          </div>
        )}
      </header>

      <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', alignItems: 'start' }}>
        
        {/* Left Column: Config & Hardware */}
        <div>
          <section className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <Cpu size={20} color="#60a5fa" />
              <h2 style={{ margin: 0 }}>Environment</h2>
            </div>
            {env ? (
               <div style={{fontSize: '0.875rem'}}>
                  <div><strong>GPU:</strong> {env.devices[0]?.name || "None"}</div>
                  <div><strong>VRAM:</strong> {env.devices[0]?.vram || "N/A"}</div>
               </div>
            ) : "Connecting..."}
          </section>

          <section className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <Settings size={20} color="#94a3b8" />
              <h2 style={{ margin: 0 }}>Training Configuration</h2>
            </div>
            
            <div className="form-group" style={{display: 'flex', flexDirection: 'column', gap: '0.75rem'}}>
              <div>
                <label className="info-label">Model ID / Path (HuggingFace or Local)</label>
                <input className="input" name="model_id" value={config.model_id} onChange={handleInputChange} style={{width: '100%'}} />
              </div>
              
              <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem'}}>
                <div>
                  <label className="info-label">Local Dataset Path (.csv / .jsonl)</label>
                  <input className="input" name="dataset_path" value={config.dataset_path || ""} onChange={handleInputChange} placeholder="e.g. data/my_train.csv" />
                </div>
                <div>
                  <label className="info-label">Max Training Rows</label>
                  <input className="input" type="number" name="dataset_len" value={config.dataset_len} onChange={handleInputChange} />
                </div>
              </div>

              <div style={{background: '#0f172a', padding: '0.75rem', borderRadius: '4px', border: '1px dashed #334155'}}>
                 <div style={{fontSize: '0.75rem', color: '#94a3b8'}}>
                   <strong>Strict Schema Requirement:</strong> Your file MUST contain three columns: <code>prompt</code>, <code>chosen</code>, and <code>rejected</code>.
                 </div>
              </div>

              <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.5rem'}}>
                 <div>
                    <label className="info-label">Batch Size</label>
                    <input className="input" type="number" name="batch_size" value={config.batch_size} onChange={handleInputChange} />
                 </div>
                 <div>
                    <label className="info-label">Epochs</label>
                    <input className="input" type="number" name="epochs" value={config.epochs} onChange={handleInputChange} />
                 </div>
                 <div>
                    <label className="info-label">Use 4-bit</label>
                    <input type="checkbox" name="use_4bit" checked={config.use_4bit} onChange={handleInputChange} />
                 </div>
              </div>

              {/* Advanced Settings Toggle */}
              <div style={{ marginTop: '0.5rem' }}>
                <button 
                  onClick={() => setIsAdvanced(!isAdvanced)} 
                  style={{ background: 'transparent', color: '#60a5fa', padding: 0, display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.875rem' }}
                >
                  {isAdvanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  {isAdvanced ? 'Hide Advanced Settings' : 'Show Advanced Settings'}
                </button>
              </div>

              {isAdvanced && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', borderTop: '1px solid #334155', paddingTop: '1rem', marginTop: '0.5rem' }}>
                  
                  {/* Optimization Group */}
                  <div>
                    <div className="info-label" style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Optimization</div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                      <div>
                        <label className="info-label">Learning Rate</label>
                        <input className="input" type="number" step="0.000001" name="learning_rate" value={config.learning_rate} onChange={handleInputChange} />
                      </div>
                      <div>
                        <label className="info-label">Grad Accumulation</label>
                        <input className="input" type="number" name="grad_accum" value={config.grad_accum} onChange={handleInputChange} />
                      </div>
                    </div>
                  </div>

                  {/* LoRA Group */}
                  <div>
                    <div className="info-label" style={{ fontWeight: 600, marginBottom: '0.5rem' }}>LoRA (Low-Rank Adaptation)</div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.5rem' }}>
                      <div>
                        <label className="info-label">Rank (R)</label>
                        <input className="input" type="number" name="lora_r" value={config.lora_r} onChange={handleInputChange} />
                      </div>
                      <div>
                        <label className="info-label">Alpha</label>
                        <input className="input" type="number" name="lora_alpha" value={config.lora_alpha} onChange={handleInputChange} />
                      </div>
                      <div>
                        <label className="info-label">Dropout</label>
                        <input className="input" type="number" step="0.01" name="lora_dropout" value={config.lora_dropout} onChange={handleInputChange} />
                      </div>
                    </div>
                  </div>

                  {/* Data Quality Group */}
                  <div>
                    <div className="info-label" style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Data Filtering</div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                      <div>
                        <label className="info-label">Max Seq Length</label>
                        <input className="input" type="number" name="max_seq_length" value={config.max_seq_length} onChange={handleInputChange} />
                      </div>
                      <div>
                        <label className="info-label">Max Resp Len</label>
                        <input className="input" type="number" name="max_response_len" value={config.max_response_len} onChange={handleInputChange} />
                      </div>
                    </div>
                  </div>

                </div>
              )}

              <button onClick={startTraining} disabled={loading} style={{marginTop: '1rem', width: '100%', background: '#059669'}}>
                {loading ? 'Launching Pipeline...' : 'Run Full RLHF Pipeline'}
              </button>
            </div>
          </section>
        </div>

        {/* Right Column: Logs & Jobs */}
        <div>
          {selectedJob && (
            <section className="card" style={{ display: 'flex', flexDirection: 'column' }}>
               <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <FileText size={20} color="#60a5fa" />
                    <h2 style={{ margin: 0 }}>Live Logs: Job #{selectedJob.id}</h2>
                  </div>
                  <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
                    {isPaused && (
                      <div style={{background: '#7f1d1d', color: '#fca5a5', padding: '0.25rem 0.75rem', borderRadius: '999px', fontSize: '0.75rem', fontWeight: 600}}>
                        PAUSED: QUALITY GATE
                      </div>
                    )}
                    <button onClick={() => setSelectedJob(null)} style={{background: 'transparent', color: '#94a3b8', padding: 0}}>Close</button>
                  </div>
               </div>

               {isPaused && (
                <div style={{background: '#1e1b4b', border: '1px solid #4338ca', padding: '1rem', borderRadius: '4px', marginBottom: '1rem'}}>
                  <div style={{fontWeight: 600, color: '#c7d2fe', marginBottom: '0.25rem'}}>⚠️ High Data Loss Detected</div>
                  <div style={{fontSize: '0.875rem', color: '#94a3b8', marginBottom: '1rem'}}>
                    Keep ratio is too low (<strong>{pausedStats}</strong>). Continue training?
                  </div>
                  <div style={{display: 'flex', gap: '0.5rem'}}>
                    <button onClick={approveJob} style={{background: '#059669', fontSize: '0.875rem', padding: '0.4rem 1rem'}}>Proceed</button>
                    <button onClick={abortJob} style={{background: '#b91c1c', fontSize: '0.875rem', padding: '0.4rem 1rem'}}>Abort</button>
                  </div>
                </div>
               )}

               <pre style={{ 
                 height: '300px',
                 background: '#0f172a', 
                 padding: '1rem', 
                 borderRadius: '4px', 
                 overflow: 'auto', 
                 fontSize: '0.75rem',
                 color: '#e2e8f0',
                 whiteSpace: 'pre-wrap'
               }}>
                 {logs || "Waiting for output..."}
               </pre>
            </section>
          )}

          <section className="card">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Activity size={20} color="#fbbf24" />
                <h2 style={{ margin: 0 }}>Recent Jobs</h2>
              </div>
              {jobs.length > 0 && (
                <button 
                  onClick={async () => {
                    if(confirm("Are you sure you want to delete ALL jobs? This will permanently wipe all logs and training outputs for every run in history.")) {
                      try {
                        await axios.delete(`${API_BASE}/jobs`);
                        setSelectedJob(null);
                      } catch (err) { alert("Failed to clear history"); }
                    }
                  }}
                  style={{padding: '0.2rem 0.5rem', fontSize: '0.75rem', background: '#7f1d1d'}}
                >
                  Clear All History
                </button>
              )}
            </div>
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Status</th>
                  <th>Actions</th>
                  <th>Management</th>
                </tr>
              </thead>
              <tbody>
                {[...jobs].reverse().slice(0, 10).map(job => (
                  <tr key={job.id}>
                    <td>#{job.id}</td>
                    <td>
                      <span className={`status-badge status-${job.status.toLowerCase()}`}>
                        {job.status}
                      </span>
                    </td>
                    <td>
                      <button onClick={() => setSelectedJob(job)} style={{padding: '0.2rem 0.5rem', fontSize: '0.75rem', marginRight: '0.25rem'}}>View Logs</button>
                      {job.status === 'COMPLETED' && (
                        <a 
                          href={`${API_BASE}/jobs/${job.id}/download`} 
                          download
                          style={{
                            padding: '0.2rem 0.5rem', 
                            fontSize: '0.75rem', 
                            background: '#059669', 
                            color: 'white', 
                            textDecoration: 'none', 
                            borderRadius: '4px',
                            display: 'inline-block'
                          }}
                        >
                          Download Outputs
                        </a>
                      )}
                    </td>
                    <td>
                      {(job.status === 'RUNNING' || job.status === 'AWAITING_APPROVAL') && (
                        <button 
                          onClick={async () => {
                            const msg = job.status === 'RUNNING' 
                              ? "Are you sure you want to stop this training job? This will kill the process immediately."
                              : "Are you sure you want to abort this job?";
                            if(confirm(msg)) {
                                try {
                                  await axios.post(`${API_BASE}/jobs/${job.id}/abort`);
                                  // The poller will pick up the status change
                                } catch (err) { alert("Failed to cancel job"); }
                            }
                          }} 
                          style={{padding: '0.2rem 0.5rem', fontSize: '0.75rem', background: '#b91c1c', marginRight: '0.25rem'}}
                        >
                          Cancel
                        </button>
                      )}
                      <button 
                        onClick={async () => {
                          if(confirm(`Are you sure you want to delete Job #${job.id}? This will remove it from history and delete its logs/outputs.`)) {
                              try {
                                await axios.delete(`${API_BASE}/jobs/${job.id}`);
                                if (selectedJob && selectedJob.id === job.id) setSelectedJob(null);
                              } catch (err) { alert(err.response?.data?.detail || "Failed to delete job"); }
                          }
                        }} 
                        disabled={job.status === 'RUNNING' || job.status === 'AWAITING_APPROVAL'}
                        title={ (job.status === 'RUNNING' || job.status === 'AWAITING_APPROVAL') ? "Cancel the job before deleting" : "Delete this job" }
                        style={{
                          padding: '0.2rem 0.5rem', 
                          fontSize: '0.75rem', 
                          background: (job.status === 'RUNNING' || job.status === 'AWAITING_APPROVAL') ? '#334155' : '#4b5563',
                          cursor: (job.status === 'RUNNING' || job.status === 'AWAITING_APPROVAL') ? 'not-allowed' : 'pointer',
                          opacity: (job.status === 'RUNNING' || job.status === 'AWAITING_APPROVAL') ? 0.5 : 1
                        }}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        .input {
          background: #0f172a;
          border: 1px solid #334155;
          color: white;
          padding: 0.4rem;
          border-radius: 4px;
          font-size: 0.9rem;
        }
        .text-green { color: #10b981; }
        .text-error { color: #ef4444; }
      `}} />
    </div>
  );
}

export default App;

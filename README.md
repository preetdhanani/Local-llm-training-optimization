# RLHF Dashboard: Standardized On-Premise Training

A professional, privacy-first RLHF (Reinforcement Learning from Human Feedback) pipeline with a modern React/FastAPI dashboard. Designed for "Local-First" training, this project ensures GDPR compliance by processing sensitive data entirely on your machine.

## 🚀 Key Features
- **Strict Data Standard**: Uses an industry-standard 3-column CSV/JSONL format.
- **Modern Dashboard**: React + Vite frontend with real-time log streaming.
- **Robust Backend**: FastAPI + SQLite job registry tracks every training run.
- **Process Isolation**: Ensures 100% VRAM recovery after every job.
- **Pre-flight Health**: Checks GPU, CUDA, and library health before training.

---

## 🛠️ Installation & Setup

### 1. Requirements
- **Hardware**: NVIDIA GPU (6GB+ VRAM recommended).
- **Software**: Python 3.10+, Node.js 18+.

### 2. Backend Setup
```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd dashboard
npm install
```

---

## 📂 Dataset Standard
This project enforces a strict local ingestion standard. Your training file must be a `.csv` or `.jsonl` with exactly these three column headers:

| Column | Description | Example Content |
| :--- | :--- | :--- |
| **prompt** | The initial query or context. | "Explain RLHF simply." |
| **chosen** | The preferred multi-turn conversation. | "Human: ... \n\nAssistant: ..." |
| **rejected** | The poor-quality response to avoid. | "I don't know." |

> **Note**: For best results, use `Human:` and `Assistant:` markers within your chosen/rejected strings for multi-turn training.

---

## 🏃 Running the Project

### Phase 1: Start the Backend
From the project root:
```bash
python -m uvicorn api.main:app --reload
```
- API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/env-check](http://localhost:8000/env-check)

### Phase 2: Start the Dashboard
In a new terminal:
```bash
cd dashboard
npm run dev
```
- Dashboard UI: [http://localhost:5173](http://localhost:5173)

---

## 🧪 Quick Smoke Test
1. Open the Dashboard.
2. In **Local Dataset Path**, enter: `test_datasets/standard_test_data.csv`.
3. Set **Max Training Rows** to `3`.
4. Click **Run Full RLHF Pipeline**.
5. Watch the live logs stream your training progress!

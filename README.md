# 🚀 RLHF Dashboard: Standardized On-Premise Training

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![React Version](https://img.shields.io/badge/react-19.0-cyan.svg)](https://react.dev/)

A professional, privacy-first RLHF (Reinforcement Learning from Human Feedback) training pipeline equipped with a modern React + Vite dashboard and FastAPI backend. Specifically engineered for **"Local-First"** environments, this framework ensures GDPR and enterprise privacy compliance by keeping training data and model weights entirely on your own infrastructure.

---

## 🏗️ System Architecture

This repository operates on a split frontend-backend architecture designed for robustness and process isolation:
1. **React Dashboard (Frontend)**: Real-time UI for configuring training parameters, submitting jobs, and live streaming execution logs.
2. **FastAPI Server (Backend)**: Orchestrates model configurations, maintains the job run database (SQLite), and schedules training.
3. **Training Engine (Isolated Process)**: Initiates isolated PyTorch processes running SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization) pipelines, freeing 100% of VRAM upon completion or termination.

---

---

## 🛠️ Local Development Setup

Run and modify the project locally with your Python and Node.js environments:

### 1. Requirements
- NVIDIA GPU is required.
- **Software**: Python 3.10+, Node.js 18+.

### 2. Backend Setup
Activate a virtual environment and install the requirements:
```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (includes PyTorch, HF transformers, TRL, etc.)
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd dashboard
npm install
cd ..
```

### 4. Running the Complete Stack (1-Shot)
To launch both the FastAPI backend and React Vite server concurrently, run the orchestrator script:
```bash
python run_dev.py
```
This utility automatically:
- Checks and installs missing dashboard dependencies.
- Runs both servers concurrently.
- Streams stdout/stderr from both processes, using prefixes (`[Backend]` / `[Frontend]`) for clarity.
- Cleans up and shuts down all child processes upon pressing `Ctrl+C`.

## 🤖 Specifying Local Model Paths

You can fine-tune both models hosted on **Hugging Face** and models stored **locally** on your machine.

### 1. In Local Development Mode (`run_dev.py`)
Simply enter the absolute or relative path to your local model folder (which must contain the standard config/model files like `config.json`, `model.safetensors`, etc.) in the **Model ID / Path** input field in the dashboard.
- *Example Absolute Path (Windows)*: `C:/models/Qwen2.5-0.5B` (always use forward slashes `/` to avoid string escaping issues).
- *Example Relative Path*: `models/Qwen2.5-0.5B`

---

## 📂 Dataset Ingestion Standard

The training pipeline enforces a strict format to prevent common dataset formatting issues. Your training dataset file must be a `.csv` or `.jsonl` file containing exactly these three columns:

| Column Header | Description | Example Content |
| :--- | :--- | :--- |
| `prompt` | The context or query given to the model. | `"Explain RLHF simply."` |
| `chosen` | The preferred high-quality response or conversation. | `"Human: ... \n\nAssistant: [high-quality response]"` |
| `rejected` | The poor-quality response to avoid. | `"Human: ... \n\nAssistant: [low-quality response]"` |

> [!NOTE]
> For best multi-turn training results, make sure `Human:` and `Assistant:` conversation markers are included in your chosen and rejected strings.

---

## 🧪 Smoke Test / Verification

To verify that your environment is fully configured and ready for training:

1. Launch the application with `run_dev.py`.
2. Open the dashboard at [http://localhost:6767](http://localhost:6767).
3. Under **Local Dataset Path**, enter the path to the standard test dataset:
   `test_datasets/standard_test_data.csv`
4. Set **Max Training Rows** to `3` (to limit the run size for testing).
5. Click **Run Full RLHF Pipeline**.
6. Observe the live logs streaming directly to the dashboard, guiding you through the SFT and DPO stages.

---

## 📁 Repository Structure

```
.
├── api/                  # FastAPI codebase
│   ├── database.py       # SQLAlchemy database engine
│   ├── env_manager.py    # GPU & environment check helper
│   ├── main.py           # API endpoints & runner isolation
│   ├── models.py         # SQLAlchemy schemas (SQLite)
│   └── schemas.py        # Pydantic payloads
├── dashboard/            # React + Vite frontend
│   ├── src/              # React components & UI logic
│   ├── nginx.conf        # Nginx config for static routing
│   └── nginx.conf        # Nginx config for static routing
├── src/                  # Core RLHF pipeline codebase
│   ├── config.py         # Model/Training configs
│   ├── pipelines/        # Pipeline orchestration (SFT -> DPO)
│   └── utils/            # Shared log and system utilities
├── run_dev.py            # Local development concurrent orchestrator
└── README.md             # This document
```

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).

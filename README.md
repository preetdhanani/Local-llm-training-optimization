# 🚀 RLHF Dashboard: Standardized On-Premise Training

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker Support](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![React Version](https://img.shields.io/badge/react-19.0-cyan.svg)](https://react.dev/)

A professional, privacy-first RLHF (Reinforcement Learning from Human Feedback) training pipeline equipped with a modern React + Vite dashboard and FastAPI backend. Specifically engineered for **"Local-First"** environments, this framework ensures GDPR and enterprise privacy compliance by keeping training data and model weights entirely on your own infrastructure.

---

## 🏗️ System Architecture

This repository operates on a split frontend-backend architecture designed for robustness and process isolation:
1. **React Dashboard (Frontend)**: Real-time UI for configuring training parameters, submitting jobs, and live streaming execution logs.
2. **FastAPI Server (Backend)**: Orchestrates model configurations, maintains the job run database (SQLite), and schedules training.
3. **Training Engine (Isolated Process)**: Initiates isolated PyTorch processes running SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization) pipelines, freeing 100% of VRAM upon completion or termination.

---

## ⚡ Quick Start: 1-Shot Docker Setup (Single Container)

We configure the React frontend to compile and build inside the image, allowing **FastAPI to serve both the API and the React website directly on port 6767** from a single Docker container.

### 1. Prerequisites
- [Docker](https://www.docker.com/get-started/) and [Docker Compose](https://docs.docker.com/compose/) installed.
- *(Optional)* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to enable GPU acceleration in Docker.

### 2. Startup Command
Run the following command in the root folder of the project:
```bash
docker compose up --build
```

- **Dashboard UI & Backend API**: [http://localhost:6767](http://localhost:6767)
- **API Documentation**: [http://localhost:6767/docs](http://localhost:6767/docs)
- **API Health Check**: [http://localhost:6767/env-check](http://localhost:6767/env-check)

> [!TIP]
> **Enabling GPUs in Docker**: If you have an NVIDIA GPU, edit the `docker-compose.yml` file and uncomment the `deploy` configuration block under the `rlhf-app` service. This exposes the host GPU drivers to the training engine.
> 
> **Model Caching**: Hugging Face models are cached inside the project's local `.cache/` folder (both in Docker and local mode). This ensures that once a model is downloaded, it is loaded locally and never downloaded again.

---

## 🛠️ Local Development Setup

If you prefer to run and modify the project locally without Docker:

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
To launch both the FastAPI backend (port 8000) and React Vite server (port 6767) concurrently, run the orchestrator script:
```bash
python run_dev.py
```
This utility automatically:
- Checks and installs missing dashboard dependencies.
- Runs both servers concurrently.
- Streams stdout/stderr from both processes, using prefixes (`[Backend]` / `[Frontend]`) for clarity.
- Cleans up and shuts down all child processes upon pressing `Ctrl+C`.

---

## 🤖 Specifying Local Model Paths

You can fine-tune both models hosted on **Hugging Face** and models stored **locally** on your machine.

### 1. In Local Development Mode (`run_dev.py`)
Simply enter the absolute or relative path to your local model folder (which must contain the standard config/model files like `config.json`, `model.safetensors`, etc.) in the **Model ID / Path** input field in the dashboard.
- *Example Absolute Path (Windows)*: `C:/models/Qwen2.5-0.5B` (always use forward slashes `/` to avoid string escaping issues).
- *Example Relative Path*: `models/Qwen2.5-0.5B`

### 2. In Docker Mode
Since containers have isolated filesystems, you must mount the host directory containing your models into the container first:
1. Open `docker-compose.yml` and add a volume mapping under the `rlhf-app` service:
   ```yaml
   volumes:
     - ./logs:/app/logs
     - ./outputs:/app/outputs
     - ./jobs.db:/app/jobs.db
     - ./.cache:/app/.cache
     - C:/path/to/your/models:/models  # Map your host models directory
   ```
2. In the Dashboard UI, specify the path relative to the container mount: `/models/Qwen2.5-0.5B`.

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

1. Launch the application (either via Docker Compose or `run_dev.py`).
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
│   ├── main.py           # API endpoints & runner isolation & static files
│   ├── models.py         # SQLAlchemy schemas (SQLite)
│   └── schemas.py        # Pydantic payloads
├── dashboard/            # React + Vite frontend
│   ├── src/              # React components & UI logic
│   └── Dockerfile        # Frontend build pipeline
├── src/                  # Core RLHF pipeline codebase
│   ├── config.py         # Model/Training configs
│   ├── pipelines/        # Pipeline orchestration (SFT -> DPO)
│   └── utils/            # Shared log and system utilities
├── Dockerfile            # Unified multi-stage Docker build config
├── docker-compose.yml    # Single service orchestrator (port 6767)
├── run_dev.py            # Local development concurrent orchestrator
└── README.md             # This document
```

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).

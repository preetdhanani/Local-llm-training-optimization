# RLHF - Hands-on 

This repository contains an end-to-end RLHF pipeline (SFT → DPO) using Transformers, TRL, PEFT/LoRA, and optional 4-bit quantization. The project provides a full-pipeline entry point plus standalone SFT and DPO runners.

**Project Entry Points**
- **Full pipeline:** `python main.py` (preprocess once → SFT → DPO)
- **SFT only:** `python -m src.pipelines.run_sft`
- **DPO only:** `python -m src.pipelines.run_dpo`

Files of interest:
- Configuration: [src/config/settings.py](src/config/settings.py)
- Full pipeline orchestration: [src/pipelines/run_full_pipeline.py](src/pipelines/run_full_pipeline.py)
- Dataset loading & preprocessing: [src/data/loaders.py](src/data/loaders.py)
- Model loader & quantization helpers: [src/models/model_loader.py](src/models/model_loader.py)
- W&B helpers: [src/utils/wandb_utils.py](src/utils/wandb_utils.py)

**Supported environment**
- Recommended: NVIDIA GPU with CUDA (GPU-first path). Project pins `torch==2.5.1+cu121` in `requirements.txt`
- Fallback: CPU-only (slower) — documented below.
- Python: 3.10.x recommended
- Internet access for Hugging Face model/dataset downloads and optional W&B

Quick disk/memory guidance
- Expect several GBs free for model & dataset cache (HF cache in `./.cache/datasets`).
- For 4-bit quantized runs a modern GPU with 16+ GB VRAM is recommended for comfortable runs; adjust `dataset_len`, `batch_size`, and `grad_accum` for low-VRAM setups.

Setup (Windows / Linux)
1. Clone the repo and change directory:

   `git clone <repo-url> && cd "s:/study/data scientist/projects/RLHF - handson- pytorch"`

2. Create and activate a virtual environment (example with venv):

   Windows:
   `python -m venv .venv`
   `.\.venv\Scripts\activate`

   Linux / macOS:
   `python -m venv .venv`
   `source .venv/bin/activate`

3. Install dependencies:

   `pip install -U pip`
   `pip install -r requirements.txt`

   Note: The repository `requirements.txt` contains GPU-pinned wheels (torch+cu121). If you need a CPU-only PyTorch, install the appropriate CPU wheel from PyTorch and then run `pip install -r requirements.txt` while skipping or removing the torch lines.

4. (Optional) Hugging Face auth for private models:

   `huggingface-cli login`

W&B setup (optional)
- If you want experiment tracking, sign up at Weights & Biases and run `wandb login` to authorize.
- W&B settings are controlled with fields in [src/config/settings.py](src/config/settings.py): `wandb_project`, `wandb_entity`, `wandb_mode`, `wandb_log_artifacts`.

Configuration
- Edit [src/config/settings.py](src/config/settings.py) to tune for your machine. Important knobs:
  - `dataset_len`: number of dataset examples to select (use small value for smoke tests)
  - `model_id`: HF model id (default: Qwen/Qwen2.5-0.5B-Instruct)
  - `batch_size`, `grad_accum`, `learning_rate`, `epochs`
  - `use_4bit`: set `False` for CPU runs or if bitsandbytes is unavailable

Using Custom Datasets
The framework is dynamic and supports loading your own data from local files (CSV, JSON, or JSONL).

1. **Prepare your data**: Ensure your file has columns for the prompt, the preferred response, and (for DPO) the rejected response.
2. **Update Settings**: In `src/config/settings.py`, configure the following:
   ```python
   dataset_type = "local"
   dataset_path = "./path/to/your/data.csv"
   dataset_format = "custom_columns" # or "standard_messages" for OpenAI format
   
   # Map your specific column names
   prompt_column = "your_question_column"
   chosen_column = "your_good_answer_column"
   rejected_column = "your_bad_answer_column"
   ```
3. **Chat Templates**: The project automatically uses the model's native chat template (via Hugging Face `apply_chat_template`), so you don't need to manually add special tokens to your data.

Running the pipeline
- Full run (recommended to validate end-to-end):

  `python main.py`

- Run SFT only (preprocess + SFT):

  `python -m src.pipelines.run_sft`

- Run DPO only (preprocess + DPO):

  `python -m src.pipelines.run_dpo`

Behavior notes
- The full pipeline runs an upfront Phase 0 preprocessing step once and reuses the processed datasets for SFT and DPO. See [src/pipelines/run_full_pipeline.py](src/pipelines/run_full_pipeline.py) for details.

Outputs & logs
- Trained models are saved under timestamped folders in `./outputs` — e.g. `./outputs/sft_output_YYYY-MM-DD_HH-MM-SS` and `./outputs/dpo_output_YYYY-MM-DD_HH-MM-SS`.
- Logs are written to the `logs/` directory. Pipeline log naming uses the helper in [src/utils/logger.py](src/utils/logger.py) and will include the phase name and timestamp.

Troubleshooting
- CUDA / torch mismatch: ensure the installed `torch` wheel matches your CUDA driver. Check `python -c "import torch; print(torch.version.cuda)"`.
- bitsandbytes failures: `bitsandbytes` requires a matching CUDA toolchain. If you see `libbitsandbytes` load errors, either install a compatible CUDA or set `use_4bit=False` in [src/config/settings.py](src/config/settings.py) and reinstall dependencies.
- Out of memory (OOM): lower `dataset_len`, reduce `batch_size`, increase `grad_accum`, or disable 4-bit and run on CPU (much slower). Also reduce `max_seq_length` in config.
- Dataset download issues: clear HF cache `rm -rf ./.cache/datasets` and retry; ensure internet access.
- W&B not logging: ensure `wandb login` was run and `wandb_project` is set in [src/config/settings.py](src/config/settings.py).

Quick start checklist
- [ ] Create and activate virtual environment
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] (Optional) `huggingface-cli login` and `wandb login`
- [ ] Edit `src/config/settings.py` for machine-specific settings (if needed)
- [ ] Run `python main.py` and confirm `./outputs` and `./logs` entries

If you want I can also add a short smoke-test mode (small `dataset_len`) and a helper script to install a CPU-only dependency set. Want me to add that?
# End-to-End-RLHF-Training-Framework

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu121
"""
Main entry point for RLHF training pipeline.

Run this file to execute the full RLHF pipeline (SFT → DPO):
    python main.py

Or run individual trainers:
    python -m src.pipelines.run_sft      # SFT only
    python -m src.pipelines.run_dpo      # DPO only (uses base model)
    python -m src.pipelines.run_full_pipeline  # Full pipeline (SFT → DPO)
"""
from src.pipelines.run_full_pipeline import run_full_pipeline


if __name__ == "__main__":
    run_full_pipeline()

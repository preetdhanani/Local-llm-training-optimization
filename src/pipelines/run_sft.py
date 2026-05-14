"""SFT training pipeline (entry point)."""
from datetime import datetime

from src.config import TrainConfig
from src.data.loaders import load_sft_dataset
from src.models.model_loader import load_model_and_tokenizer
from src.trainers.sft import train_sft
from src.utils.logger import setup_logger, get_timestamped_log_path
from src.utils.wandb_utils import (
    finish_run,
    format_group_name,
    format_run_name,
    init_wandb_run,
)


def main():
    """Run SFT training end-to-end."""
    # Setup
    log_path = get_timestamped_log_path("sft")
    logger = setup_logger("sft", log_path)
    cfg = TrainConfig()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    init_wandb_run(
        cfg,
        run_name=format_run_name("sft", cfg.model_id, timestamp),
        group=cfg.wandb_group or format_group_name(cfg.model_id, timestamp),
        job_type="sft",
        config_overrides={"pipeline_timestamp": timestamp},
    )

    logger.info(f"SFT training started. Logs saved to: {log_path}")
    
    # Load
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    logger.info("Preparing SFT dataset...")
    train_ds, eval_ds = load_sft_dataset(cfg, tokenizer)
    logger.info(f"  Train: {len(train_ds)} examples")
    logger.info(f"  Eval : {len(eval_ds)} examples")

    # Note: W&B logging is handled by trainer's native report_to="wandb"
    # Dataset stats and samples are disabled to keep W&B clean (only real-time metrics)

    # Train
    logger.info("Starting SFT training...")
    logger.info(f"  Model     : {cfg.model_id}")
    logger.info(f"  Batch size: {cfg.batch_size}")
    logger.info(f"  Learning rate: {cfg.learning_rate}")
    logger.info(f"  Epochs    : {cfg.epochs}")

    try:
        train_sft(cfg, model, tokenizer, train_ds, eval_ds)
        logger.info(f"Training complete. Model saved to {cfg.sft_output_dir}")
    finally:
        finish_run(cfg)


if __name__ == "__main__":
    main()

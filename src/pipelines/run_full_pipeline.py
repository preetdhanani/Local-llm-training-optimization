"""Full RLHF pipeline: SFT → DPO (end-to-end)."""
import os
from datetime import datetime
from typing import Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel

from src.config import TrainConfig
from src.data.loaders import load_sft_dataset, load_dpo_dataset
from src.trainers.sft import train_sft
from src.trainers.dpo import train_dpo
from src.utils.logger import setup_logger, get_timestamped_log_path
from src.utils.paths import find_latest_sft_output
from src.utils.wandb_utils import (
    finish_run,
    format_group_name,
    format_run_name,
    init_wandb_run,
)


def run_full_pipeline(cfg: Optional[TrainConfig] = None):
    """
    Execute full RLHF pipeline end-to-end.
    """
    # Initialize config if not provided
    if cfg is None:
        cfg = TrainConfig()

    # Determine base output directories
    base_outputs = getattr(cfg, "sft_output_dir", "./outputs")
    if "_" in base_outputs: # If it already has a timestamp, keep it
        base_sft_path = base_outputs
    else:
        base_sft_path = "./outputs/sft_output"

    # Generate single timestamp for entire pipeline run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg.sft_output_dir = f"{base_sft_path}_{timestamp}"
    cfg.dpo_output_dir = f"./outputs/dpo_output_{timestamp}"

    # Ensure base outputs directory exists
    try:
        os.makedirs("./outputs", exist_ok=True)
    except FileExistsError:
        if not os.path.isdir("./outputs"):
            raise
    
    run_group = cfg.wandb_group or format_group_name(cfg.model_id, timestamp)

    # ========================================
    # SET UP UNIFIED LOGGING FOR ENTIRE PIPELINE
    # ========================================
    # Create single timestamped log file for the entire pipeline run
    log_path = get_timestamped_log_path("full_pipeline")
    logger = setup_logger("full_pipeline", log_path)
    logger.info("=" * 80)
    logger.info("RLHF FULL PIPELINE START")
    logger.info("=" * 80)
    logger.info(f"Single log file: {log_path}")
    logger.info(f"SFT output will be saved to: {cfg.sft_output_dir}")
    logger.info(f"DPO output will be saved to: {cfg.dpo_output_dir}")
    logger.info("")

    # ========================================
    # PHASE 0: DATA PREPROCESSING (UPFRONT)
    # ========================================
    logger.info("=" * 80)
    logger.info("PHASE 0: DATA PREPROCESSING")
    logger.info("=" * 80)
    
    try:
        # Load base model and tokenizer (needed for both phases)
        logger.info("Loading base model and tokenizer...")
        from src.models.model_loader import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(cfg)
        logger.info(f"[OK] Model loaded: {cfg.model_id}")
        logger.info(f"[OK] Tokenizer loaded")
        logger.info("")
        
        # Load raw dataset using robust loader logic
        logger.info("Loading raw dataset...")
        if cfg.dataset_type == "local" and cfg.dataset_path:
            if not os.path.exists(cfg.dataset_path):
                raise FileNotFoundError(f"Local dataset not found at: {cfg.dataset_path}")
            ext = os.path.splitext(cfg.dataset_path)[1].lower()
            if ext == ".csv":
                raw_dataset = load_dataset("csv", data_files=cfg.dataset_path, split="train")
            elif ext in [".json", ".jsonl"]:
                raw_dataset = load_dataset("json", data_files=cfg.dataset_path, split="train")
            else:
                raise ValueError(f"Unsupported local file extension: {ext}")
            
            # Check for required columns if using custom mapping
            if cfg.dataset_format == "custom_columns":
                missing = [col for col in [cfg.prompt_column, cfg.chosen_column] if col not in raw_dataset.column_names]
                if missing:
                    raise ValueError(f"Required columns {missing} not found in dataset. Available: {raw_dataset.column_names}")
        else:
            raw_dataset = load_dataset(
                cfg.dataset_name,
                split="train",
                num_proc=1,
                keep_in_memory=False,
                download_mode="force_redownload",
            )
        
        logger.info(f"[OK] Raw dataset loaded: {len(raw_dataset)} total examples")
        logger.info("")
        
        # Preprocess for SFT
        logger.info("Preprocessing for SFT...")
        from src.data.loaders import preprocess_sft_dataset
        train_ds_sft, eval_ds_sft = preprocess_sft_dataset(cfg, tokenizer, raw_dataset, logger)
        logger.info("")
        
        # Preprocess for DPO
        logger.info("Preprocessing for DPO...")
        from src.data.loaders import preprocess_dpo_dataset
        train_ds_dpo, eval_ds_dpo = preprocess_dpo_dataset(cfg, tokenizer, raw_dataset, logger)
        logger.info("")
        
        # ========================================
        # QUALITY GATE: Check for significant data loss
        # ========================================
        total_kept = len(train_ds_dpo) + len(eval_ds_dpo)
        # Compare against the user's requested limit (dataset_len)
        requested = cfg.dataset_len
        keep_ratio = total_kept / requested if requested > 0 else 0
        
        logger.info(f"Quality Gate Check: Kept {total_kept} / {requested} (Requested) - Ratio: {keep_ratio:.1%}")
        
        if keep_ratio < 0.60:
            logger.warning("=" * 40)
            logger.warning("CRITICAL DATA LOSS DETECTED (> 40%)")
            logger.warning("The pipeline is now PAUSED and awaiting UI approval.")
            logger.warning("=" * 40)
            
            # Update database status via a signal file or direct DB access
            # For process isolation, we'll use a simple file-based signal for 'AWAITING_APPROVAL'
            # and wait for an 'approved' or 'aborted' file to appear.
            signal_dir = f"./logs/signals_{timestamp}"
            os.makedirs(signal_dir, exist_ok=True)
            with open(f"{signal_dir}/PAUSED", "w") as f:
                f.write(f"{total_kept}/{requested}")
            
            # Loop and wait for approval
            import time
            wait_start = time.time()
            approved = False
            while time.time() - wait_start < 600: # 10 minute timeout
                if os.path.exists(f"{signal_dir}/APPROVED"):
                    logger.info("[OK] Received UI Approval. Continuing training...")
                    approved = True
                    break
                if os.path.exists(f"{signal_dir}/ABORTED"):
                    logger.error("[ABORT] Received UI Abort signal. Stopping pipeline.")
                    raise RuntimeError("Pipeline aborted by user due to high data loss.")
                time.sleep(5)
            
            if not approved:
                logger.error("[TIMEOUT] No approval received within 10 minutes. Aborting.")
                raise TimeoutError("Pipeline timed out awaiting quality approval.")

        logger.info("=" * 80)
        logger.info("[OK] DATA PREPROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info("")
        
    except Exception as e:
        logger.error(f"[FAILED] Data preprocessing failed: {e}")
        logger.error("[FAILED] Aborting full pipeline")
        raise

    # ========================================
    # PHASE 1: SUPERVISED FINE-TUNING (SFT)
    # ========================================
    init_wandb_run(
        cfg,
        run_name=format_run_name("sft", cfg.model_id, timestamp),
        group=run_group,
        job_type="sft",
        config_overrides={"pipeline_timestamp": timestamp, "pipeline_phase": "sft"},
    )

    logger.info("=" * 80)
    logger.info("PHASE 1: SUPERVISED FINE-TUNING (SFT)")
    logger.info("=" * 80)

    try:
        logger.info("SFT dataset already preprocessed")
        logger.info(f"[OK] SFT Train: {len(train_ds_sft)} examples")
        logger.info(f"[OK] SFT Eval : {len(eval_ds_sft)} examples")

        # Note: W&B logging is handled by trainer's native report_to="wandb"
        # Dataset stats and samples are disabled to keep W&B clean (only real-time metrics)


        # Run SFT training
        logger.info("Starting SFT training...")
        logger.info(f"  Model         : {cfg.model_id}")
        logger.info(f"  Batch size    : {cfg.batch_size}")
        logger.info(f"  Learning rate : {cfg.learning_rate}")
        logger.info(f"  Epochs        : {cfg.epochs}")
        logger.info(f"  Output dir    : {cfg.sft_output_dir}")

        train_sft(cfg, model, tokenizer, train_ds_sft, eval_ds_sft)

        logger.info("=" * 80)
        logger.info("[OK] SFT training complete!")
        logger.info(f"[OK] SFT model saved to: {cfg.sft_output_dir}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"[FAILED] SFT training failed: {e}")
        logger.error("[FAILED] Aborting full pipeline")
        raise
    finally:
        finish_run(cfg)

    # ========================================
    # PHASE 2: DIRECT PREFERENCE OPTIMIZATION (DPO)
    # ========================================
    init_wandb_run(
        cfg,
        run_name=format_run_name("dpo", cfg.model_id, timestamp),
        group=run_group,
        job_type="dpo",
        config_overrides={"pipeline_timestamp": timestamp, "pipeline_phase": "dpo"},
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 2: DIRECT PREFERENCE OPTIMIZATION (DPO)")
    logger.info("=" * 80)

    try:
        # Automatically find latest SFT output directory
        latest_sft_dir = find_latest_sft_output("./outputs")
        logger.info(f"Found latest SFT output: {latest_sft_dir}")
        logger.info(f"Loading SFT-trained model for DPO training...")
        if not os.path.exists(latest_sft_dir):
            raise FileNotFoundError(f"SFT output directory not found: {latest_sft_dir}")

        # Load tokenizer from latest SFT checkpoint
        tokenizer = AutoTokenizer.from_pretrained(latest_sft_dir)
        logger.info("[OK] Tokenizer loaded from SFT checkpoint")

        # Load base model WITHOUT device_map to avoid meta tensor issues with PEFT
        logger.info(f"Loading base model: {cfg.model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.bfloat16,
        )
        logger.info("[OK] Base model loaded")

        # Load LoRA adapter from latest SFT training
        logger.info(f"Loading LoRA adapter from: {latest_sft_dir}")
        model = PeftModel.from_pretrained(model, latest_sft_dir)
        logger.info("[OK] LoRA adapter loaded - SFT model ready for DPO")

        # Create reference model from SFT checkpoint for DPO KL divergence computation
        from copy import deepcopy
        ref_model = deepcopy(model)
        logger.info("[OK] Reference model created (frozen copy of SFT model)")

        logger.info("DPO dataset already preprocessed")
        logger.info(f"[OK] DPO Train: {len(train_ds_dpo)} examples")
        logger.info(f"[OK] DPO Eval : {len(eval_ds_dpo)} examples")

        # Note: W&B logging is handled by trainer's native report_to="wandb"
        # Dataset stats and samples are disabled to keep W&B clean (only real-time metrics)

        # Run DPO training
        logger.info("Starting DPO training...")
        logger.info(f"  Base model    : {latest_sft_dir}")
        logger.info(f"  Ref model     : SFT checkpoint (frozen)")
        logger.info(f"  Batch size    : {cfg.batch_size}")
        logger.info(f"  Learning rate : {cfg.learning_rate}")
        logger.info(f"  Beta (DPO)    : {cfg.beta}")
        logger.info(f"  Epochs        : {cfg.epochs}")
        logger.info(f"  Output dir    : {cfg.dpo_output_dir}")

        train_dpo(cfg, model, ref_model, tokenizer, train_ds_dpo, eval_ds_dpo)

        logger.info("=" * 80)
        logger.info("[OK] DPO training complete!")
        logger.info(f"[OK] Final model saved to: {cfg.dpo_output_dir}")
        logger.info("=" * 80)
        logger.info("")
        logger.info("RLHF FULL PIPELINE COMPLETE!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"[FAILED] DPO training failed: {e}")
        logger.error("[FAILED] Aborting full pipeline")
        raise
    finally:
        finish_run(cfg)

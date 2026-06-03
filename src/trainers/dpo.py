"""DPO training orchestration."""
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from torch import nn
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer

from src.config import TrainConfig
from src.models import build_lora_config


def train_dpo(
    cfg: TrainConfig,
    model: nn.Module,
    ref_model: nn.Module = None,
    tokenizer: AutoTokenizer = None,
    train_dataset: Dataset = None,
    eval_dataset: Dataset = None,
) -> None:
    """
    Execute DPO training.

    Args:
        cfg: TrainConfig instance
        model: Loaded model (will be fine-tuned during DPO)
        ref_model: Reference model for KL divergence computation (frozen). If None, uses internal reference.
        tokenizer: Tokenizer
        train_dataset: Training dataset with "prompt", "chosen", "rejected" fields
        eval_dataset: Evaluation dataset with same fields
    """
    import sys
    
    # **DIAGNOSTIC: Check for None values BEFORE training**
    sys.stderr.write("[DEBUG DPO] Checking for None/empty values in datasets...\n")
    sys.stderr.flush()
    
    none_count = 0
    for i, row in enumerate(train_dataset):
        for key in ["prompt", "chosen", "rejected"]:
            val = row.get(key)
            if val is None:
                sys.stderr.write(f"[ERROR] Train row {i}: {key}=None\n")
                none_count += 1
            elif isinstance(val, str) and len(val) == 0:
                sys.stderr.write(f"[ERROR] Train row {i}: {key}=empty string\n")
                none_count += 1
    
    if none_count > 0:
        raise ValueError(f"DPO train dataset has {none_count} problematic fields. Cannot proceed.")
    
    sys.stderr.write(f"[DEBUG DPO] Train dataset OK: {len(train_dataset)} rows\n")
    sys.stderr.flush()
    
    # Prepare model for kbit training (if quantized)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,
    )
    model.enable_input_require_grads()

    # Build training config
    dpo_config_kwargs = {
        "output_dir": cfg.dpo_output_dir,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.epochs,
        "beta": cfg.beta,
        "max_prompt_length": cfg.max_prompt_length,
        "max_length": cfg.max_seq_length,
        "max_grad_norm": cfg.max_grad_norm,
        "warmup_ratio": cfg.dpo_warmup_ratio,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_steps": cfg.dpo_save_steps,
        "evaluation_strategy": "no",
        "eval_steps": cfg.dpo_eval_steps,
        "load_best_model_at_end": False,
        "metric_for_best_model": "eval_loss",
        "bf16": False,  # DPO with 4bit uses float16
        "gradient_checkpointing": False,
        "remove_unused_columns": False,
        "report_to": "wandb" if (cfg.wandb_project and cfg.wandb_mode != "disabled") else "none",
        "run_name": f"dpo-{cfg.model_id}",
    }
    
    # If using explicit ref_model with PEFT, need to force its use
    if ref_model is not None:
        dpo_config_kwargs["force_use_ref_model"] = True
    
    dpo_config = DPOConfig(**dpo_config_kwargs)

    # Build LoRA config
    lora_config = build_lora_config(
        r=cfg.lora_r,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
    )

    # Train
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        beta=cfg.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(cfg.dpo_output_dir)

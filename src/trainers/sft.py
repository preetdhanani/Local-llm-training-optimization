"""SFT training orchestration."""
from datasets import Dataset
from peft import LoraConfig
from torch import nn
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.config import TrainConfig
from src.models import build_lora_config


def train_sft(
    cfg: TrainConfig,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> None:
    """
    Execute SFT training.

    Args:
        cfg: TrainConfig instance
        model: Loaded model (quantized or not)
        tokenizer: Tokenizer
        train_dataset: Training dataset with "text" field
        eval_dataset: Evaluation dataset with "text" field
    """
    import gc
    import torch
    
    # Release cached GPU memory before loading dataset / setting up training
    gc.collect()
    torch.cuda.empty_cache()

    # Dynamically select precision and optimizer based on GPU capabilities
    device_is_cuda = torch.cuda.is_available()
    bf16_supported = device_is_cuda and torch.cuda.is_bf16_supported()
    fp16_supported = device_is_cuda and not bf16_supported
    optim_algo = "paged_adamw_32bit" if device_is_cuda else "adamw_torch"

    # Build training config
    sft_config = SFTConfig(
        output_dir=cfg.sft_output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,  # Prevent default batch size of 8 from OOM-ing
        gradient_accumulation_steps=cfg.grad_accum,
        eval_accumulation_steps=1,                 # Offload evaluation predictions to CPU progressively
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.sft_warmup_ratio,
        max_seq_length=cfg.max_seq_length,
        packing=False,
        dataset_text_field="text",
        bf16=bf16_supported,
        fp16=fp16_supported,
        gradient_checkpointing=True,
        optim=optim_algo,
        logging_steps=1,
        save_steps=cfg.sft_save_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.sft_eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb" if (cfg.wandb_project and cfg.wandb_mode != "disabled") else "none",
        run_name=f"sft-{cfg.model_id}",
    )

    # Build LoRA config
    lora_config = build_lora_config(
        r=cfg.lora_r,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
    )

    # Local metrics callback
    callbacks = []
    if getattr(cfg, "metrics_filepath", None):
        from src.utils.metrics import LocalMetricsCallback
        callbacks.append(LocalMetricsCallback(cfg.metrics_filepath, phase="sft"))

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(cfg.sft_output_dir)


from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
 

import config as cfg 
import os

  
 
# =========================
# DATA PROCESSING
# =========================
 
def safe_split(text: Optional[str]):
    if not isinstance(text, str):
        return None, None
 
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler("./logs/dpo_train.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("dpo_train")


# =========================
# DATA PROCESSING
# =========================

def safe_split(text: Optional[str]):
    logger.info(f"Splitting text: {text}")
    if not isinstance(text, str):
        return None, None

    parts = text.rsplit("Assistant:", 1)
    if len(parts) != 2:
        return None, None

    prompt = parts[0].strip()
    response = parts[1].strip()
    return prompt, response


def format_row(example):
    logger.info(f"Formatting example: {example[:2]}")
    cp, cr = safe_split(example.get("chosen"))
    rp, rr = safe_split(example.get("rejected"))

    valid = True

    if cp is None or rp is None or cp != rp:
        valid = False

    if cr is None or rr is None:
        valid = False

    logger.info(f"Formatted example: {example[:2]}")
    return {
        "prompt": cp if valid else "",
        "chosen": cr if valid else "",
        "rejected": rr if valid else "",
        "valid": valid,
    }


def advanced_filter(example, tokenizer):
    logger.info(f"Filtering example: {example[:2]}")
    if not example["valid"]:
        return False

    p = example["prompt"]
    c = example["chosen"]
    r = example["rejected"]

    if not p or not c or not r:
        return False

    if len(p) < cfg.min_prompt_len:
        return False
    if len(c) < cfg.min_response_len or len(r) < cfg.min_response_len:
        return False
    if len(c) > cfg.max_response_len or len(r) > cfg.max_response_len:
        return False

    if abs(len(c) - len(r)) > cfg.max_length_diff:
        return False

    p_len = len(tokenizer(p)["input_ids"])
    c_len = len(tokenizer(c)["input_ids"])
    r_len = len(tokenizer(r)["input_ids"])

    if p_len > cfg.max_prompt_length:
        return False
    if c_len > cfg.max_length or r_len > cfg.max_length:
        return False

    return True


def dataset_stats(ds):
    logger.info(f"Calculating dataset stats for {len(ds)} samples")
    lengths_c = [len(x["chosen"]) for x in ds]
    lengths_r = [len(x["rejected"]) for x in ds]
    margins = [len(x["chosen"]) - len(x["rejected"]) for x in ds]

    logger.info("Dataset stats")
    logger.info("  Samples            : %s", len(ds))
    logger.info("  Avg chosen length  : %.1f", sum(lengths_c) / len(lengths_c))
    logger.info("  Avg rejected length: %.1f", sum(lengths_r) / len(lengths_r))
    logger.info("  Avg length diff    : %.1f", sum(abs(m) for m in margins) / len(margins))
    logger.info("  Max chosen length  : %s", max(lengths_c))
    logger.info("  Min chosen length  : %s", min(lengths_c))


def load_data(tokenizer):
    logger.info(f"Loading dataset: {cfg.dataset_name}")
    ds = load_dataset(cfg.dataset_name, split="train")

    logger.info("Formatting dataset")
    ds = ds.map(format_row, remove_columns=ds.column_names)

    logger.info("Filtering dataset")
    ds = ds.filter(lambda x: advanced_filter(x, tokenizer))

    ds = ds.remove_columns(["valid"])

    logger.info("Final dataset size: %s", len(ds))
    dataset_stats(ds)
    logger.info("Sample: %s", ds[0])

    split = ds.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]


# =========================
# MODEL
# =========================

def load_model():
    logger.info(f"Loading model: {cfg.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = None
    if cfg.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,
    )
    model.enable_input_require_grads()

    return model, tokenizer


def get_lora():
    logger.info("Creating LoRA config")
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# =========================
# TRAINING
# =========================

def train():
    logger.info("Starting DPO training_______")
    model, tokenizer = load_model()
    train_ds, eval_ds = load_data(tokenizer)

    dpo_config = DPOConfig(
        output_dir=cfg.output_dir,
        remove_unused_columns=False,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        beta=cfg.beta,
        max_prompt_length=cfg.max_prompt_length,
        max_length=cfg.max_length,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        evaluation_strategy="steps",
        bf16=False,
        gradient_checkpointing=False,
        report_to="wandb",
        run_name="dpo-qwen-v2-fixed",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=get_lora(),
    )

    logger.info("Starting training")
    logger.info("  beta          : %s", cfg.beta)
    logger.info("  learning rate : %s", cfg.lr)
    logger.info("  max_grad_norm : 1.0")
    logger.info("  LoRA rank     : 16")
    logger.info("  LoRA targets  : q,k,v,o + gate,up,down")
    logger.info("  warmup ratio  : 10%%")
    logger.info("  lr scheduler  : cosine")

    trainer.train()

    logger.info("Saving model")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info("Model saved to %s", cfg.output_dir)


if __name__ == "__main__":
    train()

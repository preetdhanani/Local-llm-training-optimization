"""High-level dataset loading APIs."""
import os
import sys
import logging
from typing import Tuple, Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from src.config import TrainConfig
from src.data.filters import dataset_stats, filter_by_length, remove_duplicates
from src.data.formatters import format_for_sft, format_preference_pair

# Set HF cache directory for dataset caching
os.environ["HF_DATASETS_CACHE"] = "./.cache/datasets"


def preprocess_sft_dataset(
    cfg: TrainConfig,
    tokenizer: AutoTokenizer,
    raw_dataset: Dataset,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dataset, Dataset]:
    """Preprocess raw dataset for SFT training with logging."""
    if logger:
        logger.info(f"[SFT] Selecting {cfg.dataset_len} examples from {len(raw_dataset)} total...")
    
    ds = raw_dataset.select(range(min(cfg.dataset_len, len(raw_dataset))))
    
    if logger:
        logger.info(f"[SFT] Formatting SFT data ({len(ds)} examples)...")
    
    ds = ds.map(lambda x: format_for_sft(x, tokenizer, cfg), desc="Formatting SFT data")
    
    if logger:
        logger.info(f"[SFT] Filtering SFT data (min length > 50)...")
    
    ds = ds.filter(lambda x: len(x.get("text", "")) > 50, desc="Filtering SFT data")
    
    if logger:
        logger.info(f"[SFT] After filtering: {len(ds)} examples")
        logger.info(f"[SFT] Splitting into train/eval (90/10)...")
    
    split = ds.train_test_split(test_size=0.1, seed=cfg.seed)
    
    if logger:
        logger.info(f"[SFT] Final: Train={len(split['train'])}, Eval={len(split['test'])}")
    
    return split["train"], split["test"]


def load_sft_dataset(cfg, tokenizer):
    """Load and prepare SFT dataset."""
    if cfg.dataset_type == "local" and cfg.dataset_path:
        if not os.path.exists(cfg.dataset_path):
            raise FileNotFoundError(f"Local dataset not found at: {cfg.dataset_path}")
        
        # Determine format from extension
        ext = os.path.splitext(cfg.dataset_path)[1].lower()
        try:
            if ext == ".csv":
                ds = load_dataset("csv", data_files=cfg.dataset_path, split="train")
            elif ext in [".json", ".jsonl"]:
                ds = load_dataset("json", data_files=cfg.dataset_path, split="train")
            else:
                raise ValueError(f"Unsupported local file extension: {ext}")
        except Exception as e:
            raise RuntimeError(f"Failed to load local {ext} dataset: {e}")

        # Check for required columns if using custom mapping
        if cfg.dataset_format == "custom_columns":
            missing = [col for col in [cfg.prompt_column, cfg.chosen_column] if col not in ds.column_names]
            if missing:
                raise ValueError(f"Required columns {missing} not found in dataset. Available: {ds.column_names}")
    else:
        try:
            ds = load_dataset(
                cfg.dataset_name,
                split="train",
                num_proc=1,
                keep_in_memory=False,
                download_mode="force_redownload",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset '{cfg.dataset_name}': {e}")

    return preprocess_sft_dataset(cfg, tokenizer, ds)



def preprocess_dpo_dataset(
    cfg: TrainConfig,
    tokenizer: AutoTokenizer,
    raw_dataset: Dataset,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Preprocess raw dataset for DPO training with logging.
    """
    from src.data.dataset_cleaner import clean_preference_dataset
    
    if logger:
        logger.info(f"[DPO] Selecting {cfg.dataset_len} examples from {len(raw_dataset)} total...")
    
    ds = raw_dataset.select(range(min(cfg.dataset_len, len(raw_dataset))))
    
    # In dynamic mode, cleaning might need adjustment but we'll try to keep it robust
    if logger:
        logger.info(f"[DPO] Cleaning preference dataset ({len(ds)} examples)...")
    
    # For custom datasets, we skip the complex hh-rlhf cleaner and do simple dedup/length filters
    if cfg.dataset_type == "local" or cfg.dataset_format == "custom_columns":
         if logger: logger.info("[DPO] Using simple cleaning for custom dataset")
         # Simple dedup based on prompt/chosen
         ds = ds.map(lambda x: {"fp": f"{x.get(cfg.prompt_column)}|||{x.get(cfg.chosen_column)}"})
         ds = ds.filter(lambda x, idx: x["fp"] not in ds.select(range(idx))["fp"] if idx > 0 else True, with_indices=True)
         ds = ds.remove_columns(["fp"])
    else:
        ds = clean_preference_dataset(ds, tokenizer, verbose=False)
    
    if logger:
        logger.info(f"[DPO] Formatting preference pairs...")
    
    ds = ds.map(
        lambda x: format_preference_pair(x, cfg),
        remove_columns=ds.column_names,
        desc="Formatting DPO data"
    )
    
    before_none = len(ds)
    ds = ds.filter(
        lambda x: (
            x.get("prompt") is not None and len(str(x.get("prompt", ""))) > 0 and
            x.get("chosen") is not None and len(str(x.get("chosen", ""))) > 0 and
            x.get("rejected") is not None and len(str(x.get("rejected", ""))) > 0
        ),
        desc="Removing None/empty fields"
    )
    
    if logger:
        logger.info(f"[DPO] After None/empty filter: removed {before_none - len(ds)}, kept {len(ds)}")
        logger.info(f"[DPO] Filtering by length constraints...")
    
    limits = {
        "min_prompt_len": cfg.min_prompt_len,
        "min_response_len": cfg.min_response_len,
        "max_response_len": cfg.max_response_len,
        "max_prompt_tokens": cfg.max_prompt_length,
        "max_response_tokens": cfg.max_seq_length,
        "max_length_diff": cfg.max_length_diff,
    }
    
    ds = ds.filter(
        lambda x: filter_by_length(x, tokenizer, limits),
        desc="Filtering DPO data"
    )
    
    if "valid" in ds.column_names:
        ds = ds.remove_columns(["valid"])
    
    if logger:
        stats = dataset_stats(ds)
        logger.info(f"[DPO] After length filter: {len(ds)} examples")
        logger.info(f"[DPO] Stats - avg_chosen={stats.get('avg_chosen_length', 0):.1f}, "
                   f"avg_rejected={stats.get('avg_rejected_length', 0):.1f}, "
                   f"avg_diff={stats.get('avg_length_diff', 0):.1f}")
        logger.info(f"[DPO] Splitting into train/eval (95/5)...")
    
    split = ds.train_test_split(test_size=0.05, seed=cfg.seed)
    
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    train_dataset.set_format("torch", columns=["prompt", "chosen", "rejected"])
    eval_dataset.set_format("torch", columns=["prompt", "chosen", "rejected"])
    
    if logger:
        logger.info(f"[DPO] Final: Train={len(train_dataset)}, Eval={len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def load_dpo_dataset(
    cfg: TrainConfig,
    tokenizer: AutoTokenizer,
) -> Tuple[Dataset, Dataset]:
    """
    Load and prepare dataset for DPO training.
    """
    if cfg.dataset_type == "local" and cfg.dataset_path:
        if not os.path.exists(cfg.dataset_path):
            raise FileNotFoundError(f"Local dataset not found at: {cfg.dataset_path}")

        ext = os.path.splitext(cfg.dataset_path)[1].lower()
        try:
            if ext == ".csv":
                ds = load_dataset("csv", data_files=cfg.dataset_path, split="train")
            elif ext in [".json", ".jsonl"]:
                ds = load_dataset("json", data_files=cfg.dataset_path, split="train")
            else:
                raise ValueError(f"Unsupported local file extension: {ext}")
        except Exception as e:
            raise RuntimeError(f"Failed to load local {ext} dataset: {e}")

        # Check for required columns if using custom mapping
        if cfg.dataset_format == "custom_columns":
            missing = [col for col in [cfg.prompt_column, cfg.chosen_column, cfg.rejected_column] if col not in ds.column_names]
            if missing:
                raise ValueError(f"Required DPO columns {missing} not found in dataset. Available: {ds.column_names}")
    else:
        try:
            dataset_config = getattr(cfg, 'dataset_config', None)
            if dataset_config:
                ds = load_dataset(cfg.dataset_name, dataset_config, split="train")
            else:
                ds = load_dataset(cfg.dataset_name, split="train", keep_in_memory=False)
        except Exception as e:
            raise RuntimeError(f"Failed to download DPO dataset '{cfg.dataset_name}': {e}")

    return preprocess_dpo_dataset(cfg, tokenizer, ds)


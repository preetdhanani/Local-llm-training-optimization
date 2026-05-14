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
    
    ds = ds.map(lambda x: format_for_sft(x, tokenizer), desc="Formatting SFT data")
    
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
    """Load and prepare SFT dataset (for backward compatibility)."""
    ds = load_dataset(
        cfg.dataset_name,
        split="train",
        num_proc=1,
        keep_in_memory=False,
        download_mode="force_redownload",
    )
    return preprocess_sft_dataset(cfg, tokenizer, ds)



def preprocess_dpo_dataset(
    cfg: TrainConfig,
    tokenizer: AutoTokenizer,
    raw_dataset: Dataset,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Preprocess raw dataset for DPO training with logging.

    Steps:
      1. Select subset
      2. Clean preference dataset
      3. Format preference pairs
      4. Remove None/empty values
      5. Remove duplicates
      6. Filter by token length and character length
      7. Split into train/eval

    Args:
        cfg: TrainConfig instance
        tokenizer: Tokenizer for token counting
        raw_dataset: Raw dataset from HuggingFace
        logger: Optional logger for logging preprocessing steps

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    from src.data.dataset_cleaner import clean_preference_dataset
    
    if logger:
        logger.info(f"[DPO] Selecting {cfg.dataset_len} examples from {len(raw_dataset)} total...")
    
    ds = raw_dataset.select(range(min(cfg.dataset_len, len(raw_dataset))))
    
    if logger:
        logger.info(f"[DPO] Cleaning preference dataset ({len(ds)} examples)...")
    
    ds = clean_preference_dataset(ds, tokenizer, verbose=False)
    
    if logger:
        logger.info(f"[DPO] Formatting preference pairs...")
    
    ds = ds.map(
        format_preference_pair,
        remove_columns=ds.column_names,
        desc="Formatting DPO data"
    )
    
    before_none = len(ds)
    ds = ds.filter(
        lambda x: (
            x.get("prompt") is not None and len(x.get("prompt", "")) > 0 and
            x.get("chosen") is not None and len(x.get("chosen", "")) > 0 and
            x.get("rejected") is not None and len(x.get("rejected", "")) > 0
        ),
        desc="Removing None/empty fields"
    )
    
    if logger:
        logger.info(f"[DPO] After None/empty filter: removed {before_none - len(ds)}, kept {len(ds)}")
        logger.info(f"[DPO] Removing duplicates...")
    
    ds = remove_duplicates(ds)
    
    if logger:
        logger.info(f"[DPO] After dedup: {len(ds)} examples")
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
    Load and prepare dataset for DPO training (for backward compatibility).

    Args:
        cfg: TrainConfig instance
        tokenizer: Tokenizer for token counting

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    dataset_config = getattr(cfg, 'dataset_config', None)
    if dataset_config:
        ds = load_dataset(cfg.dataset_name, dataset_config, split="train")
    else:
        ds = load_dataset(cfg.dataset_name, split="train", keep_in_memory=False)

    return preprocess_dpo_dataset(cfg, tokenizer, ds)


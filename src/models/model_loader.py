"""Model and tokenizer loading."""
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import TrainConfig
from src.models.quantization import get_bnb_config


def load_model_and_tokenizer(
    cfg: TrainConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer with optional quantization.

    Args:
        cfg: TrainConfig instance

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Build model kwargs
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }

    # Add quantization if enabled
    bnb_config = get_bnb_config(use_4bit=cfg.use_4bit)
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        **model_kwargs,
    )

    # Set pad token in config
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

"""Quantization configuration for 4-bit/8-bit training."""
from typing import Optional

import torch
from transformers import BitsAndBytesConfig


def get_bnb_config(
    use_4bit: bool = True,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Optional[BitsAndBytesConfig]:
    """
    Create BitsAndBytes quantization config.

    Args:
        use_4bit: If True, use 4-bit quantization; if False, return None
        compute_dtype: Compute dtype (torch.bfloat16, torch.float16, etc.)

    Returns:
        BitsAndBytesConfig or None
    """
    if not use_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

"""LoRA configuration factory."""
from peft import LoraConfig, TaskType


def build_lora_config(
    r: int = 16,
    alpha: int = 32,
    target_modules: list = None,
    dropout: float = 0.05,
) -> LoraConfig:
    """
    Build LoRA configuration.

    Args:
        r: LoRA rank
        alpha: LoRA scaling factor (usually 2*r)
        target_modules: Which modules to apply LoRA to
        dropout: LoRA dropout

    Returns:
        LoraConfig instance
    """
    if target_modules is None:
        # Default: attention modules + MLP modules
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention
            "gate_proj", "up_proj", "down_proj",  # MLP
        ]

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

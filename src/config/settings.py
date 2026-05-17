"""Centralized configuration for all training."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """Unified training configuration."""
    # Dataset settings
    dataset_type: str = "huggingface"  # "huggingface" | "local"
    dataset_path: Optional[str] = None  # Path to local CSV/JSONL if type is "local"
    dataset_format: str = "custom_columns"  # "standard_messages" | "custom_columns"
    
    # Column mapping (used if dataset_format == "custom_columns")
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    system_prompt: Optional[str] = "You are a helpful assistant."

    # Dataset len
    dataset_len: int = 1000

    # Model & Dataset
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "Anthropic/hh-rlhf"
    dataset_config: Optional[str] = None  # No config needed for hh-rlhf

    # Output (timestamped at runtime in pipeline)
    sft_output_dir: str = "./outputs/sft_output"
    dpo_output_dir: str = "./outputs/dpo_output"

    # Training hyperparameters
    batch_size: int = 1
    grad_accum: int = 2
    learning_rate: float = 5e-6
    epochs: float = 1.0
    beta: float = 0.3  # DPO preference signal strength

    # Sequence lengths
    max_prompt_length: int = 256
    max_seq_length: int = 384

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Quantization
    use_4bit: bool = True

    # SFT-specific
    sft_warmup_ratio: float = 0.03
    sft_eval_steps: int = 10
    sft_save_steps: int = 100

    # DPO-specific
    dpo_warmup_ratio: float = 0.1
    dpo_eval_steps: int = 10
    dpo_save_steps: int = 100
    max_grad_norm: float = 1.0

    # Report to wandb
    report_to="wandb"

    # Data filtering (DPO)
    min_prompt_len: int = 10
    min_response_len: int = 3
    max_response_len: int = 300
    max_length_diff: int = 100

    # Other
    seed: int = 42
    log_level: str = "INFO"
    wandb_project: Optional[str] = "rlhf-handson-pytorch"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"  # online | offline | disabled
    wandb_tags: Optional[str] = None  # Comma-separated tags
    wandb_notes: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_log_artifacts: bool = False
    wandb_log_dataset_stats: bool = False
    wandb_log_samples: bool = False
    wandb_samples_count: int = 5

    def __post_init__(self):
        """Validate config values."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.lora_r <= 0:
            raise ValueError("lora_r must be positive")
        if self.wandb_samples_count <= 0:
            raise ValueError("wandb_samples_count must be positive")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be one of: online, offline, disabled")

from pydantic import BaseModel, Field
from typing import Optional, List, Any
import datetime

class JobBase(BaseModel):
    type: str
    config: dict

class JobCreate(JobBase):
    pass

class JobResponse(JobBase):
    id: int
    status: str
    pid: Optional[int]
    log_path: Optional[str]
    created_at: datetime.datetime
    updated_at: datetime.datetime

    class Config:
        from_attributes = True

class EnvResponse(BaseModel):
    status: str
    python_version: str
    torch_installed: bool
    cuda_available: bool
    device_count: int
    devices: List[dict]
    bitsandbytes_available: bool
    errors: List[str]

class TrainingConfigSchema(BaseModel):
    # Dataset settings
    dataset_path: str # Always required local path
    dataset_len: int = 1000
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Training hyperparameters
    batch_size: int = 1
    grad_accum: int = 2
    learning_rate: float = 5e-6
    epochs: float = 1.0
    beta: float = 0.3
    
    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Sequence lengths
    max_prompt_length: int = 256
    max_seq_length: int = 384
    
    # Data filtering (DPO)
    min_prompt_len: int = 10
    min_response_len: int = 3
    max_response_len: int = 300
    max_length_diff: int = 100

    # Quantization
    use_4bit: bool = True
    
    # WandB
    wandb_mode: str = "disabled"
    wandb_project: str = "rlhf-handson-pytorch"

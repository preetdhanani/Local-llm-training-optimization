# Model

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
dataset = "Anthropic/hh-rlhf"
row = 1000
saved_model: str = "./sft_final"
dpo_output_dir: str = "./dpo_output"

batch_size: int = 4
grad_accum: int = 8

lr: float = 5e-6        
epochs: float = 1.0
beta: float = 0.3      


max_prompt_length: int = 256
max_length: int = 512
 
use_4bit: bool = True
 
    # data constraints
min_prompt_len: int = 10
min_response_len: int = 3
max_response_len: int = 300
max_length_diff: int = 100  # tightened from 150 for better length-bias control
 

save_steps=200,
eval_steps=200,
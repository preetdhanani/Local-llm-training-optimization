from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType
import torch
import logging
import config
from config import MODEL_ID
from quantize_training import bnb_config
from config import dataset
from config import row
from config import batch_size
from config import grad_accum



logging.basicConfig(
    filename="logs/sft.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("sft_logger")

logger.info("Starting SFT training...")
USE_QUANTIZATION = True


model_kwargs = {
    "device_map": "auto",
    "torch_dtype": torch.bfloat16,
}
if USE_QUANTIZATION:
    model_kwargs["quantization_config"] = bnb_config
    logger.info("Using quantization for model loading.")


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    **model_kwargs,
)


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "right"
logger.info(f"Tokenizer pad token: {tokenizer.pad_token}, eos token: {tokenizer.eos_token}")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# LoRA LoraConfig
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    )
logger.info(f"LoRA config: {lora_cfg}")

dataset = load_dataset(dataset, split="train")
dataset = dataset.shuffle(seed=42).select(range(row))
logger.info(f"Loaded dataset with {len(dataset)} examples.")


def format_for_sft(example):
    # Parse dialog into turns
    text = example["chosen"]
    turns = text.strip().split("\n\n")

    messages =[]
    for turn in turns:
        if turn.startswith("Human:"):
            messages.append({"role": "user", "content" : turn[6:].strip()})
        elif turn.startswith("Assistant:"):
            messages.append({"role": "assistant", "content": turn[10:].strip()})

    formatted = tokenizer.apply_chat_template(
        messages,   
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": formatted}
    

dataset = dataset.map(format_for_sft)   
dataset = dataset.filter(lambda x: len(x["text"]) > 50) 

# Split: 90% train, 10% eval
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
logger.info(f"Train dataset size: {len(train_test_split['train'])}, Eval dataset size: {len(train_test_split['test'])}")
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]



# Training config  

sft_cfg = SFTConfig(
    output_dir="./sft_output",

    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,

    num_train_epochs=config.epochs,
    learning_rate=config.lr,
    warmup_ratio=0.03,

    max_seq_length=512,
    packing=True,
    dataset_text_field="text",

    bf16=True,
    gradient_checkpointing=True,  

    logging_steps=10,
    save_steps=200,
    evaluation_strategy="steps",
    eval_steps=100,

    report_to="wandb",
    run_name=f"sft-hh-rlhf-{MODEL_ID}" 
)

logger.info(f"SFT training config: {sft_cfg}")

# Train  
# trainer = SFTTrainer(
#     model = model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     peft_config=lora_cfg,
#     args=sft_cfg
# )

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  
    peft_config=lora_cfg,
    args=sft_cfg
)
logger.info("Initialized SFTTrainer, starting training...")

logger.info("Training started...")

trainer.train()
logger.info("Training completed, saving model and tokenizer...")
trainer.save_model("./sft_final")
logger.info("Model saved to ./sft_final")  
tokenizer.save_pretrained("./sft_final")
logger.info("Tokenizer saved to ./sft_final")
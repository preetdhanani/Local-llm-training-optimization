# Local Models Directory

This directory is designed to store local Large Language Model (LLM) weights (e.g., downloaded from Hugging Face or fine-tuned checkpoints) for local-first training. 

By placing models in this directory, you can run training runs in 100% offline mode without needing to download large weights repeatedly.

---

## Expected Directory Structure

To use a local model, download the model files manually from Hugging Face and paste them into a subfolder inside this directory. 

Each model must have its own subfolder containing standard configuration and weights files (`config.json`, model weights like `model.safetensors` or `pytorch_model.bin`, `tokenizer.json`, etc.).

Example layout for **Qwen 2.5 0.5B Instruct**:
```text
models/
└── Qwen2.5-0.5B-Instruct/
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.json
```

---

## How to Reference via Relative Path

When launching a training job through the Dashboard UI, specify the path to your model relative to the repository root:

- **Dashboard UI Field (Model ID / Path)**:
  ```text
  models/Qwen2.5-0.5B-Instruct
  ```

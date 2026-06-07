# Dataset Transformation Utilities

This directory contains standalone scripts to convert public RLHF/DPO datasets into the CSV format expected by this project.

## Scripts

### 1. Anthropic HH-RLHF Transform (`transform_hh_rlhf.py`)
Fetches the `Anthropic/hh-rlhf` dataset from Hugging Face, extracts the initial user query (prompt), and structures it into columns (`prompt`, `chosen`, `rejected`).

**How to run:**
```bash
python test_datasets/transform/transform_hh_rlhf.py
```

**Output:**
Saves the formatted dataset directly to:
`test_datasets/hh_rlhf_dpo.csv`

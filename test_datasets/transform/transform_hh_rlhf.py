from datasets import load_dataset
import pandas as pd
import re
import os

# Load Anthropic HH-RLHF
print("Loading Anthropic/hh-rlhf dataset from Hugging Face...")
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

def extract_first_human_message(text):
    """
    Extract the first Human question.
    """
    match = re.search(r"Human:\s*(.*?)\n\nAssistant:", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"Human:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return ""

rows = []

print("Extracting prompt, chosen, and rejected fields...")
for row in dataset:
    chosen = row["chosen"].strip()
    rejected = row["rejected"].strip()

    prompt = extract_first_human_message(chosen)

    rows.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    })

df = pd.DataFrame(rows)

# Save directly to the test_datasets/ parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
output_file = os.path.join(parent_dir, "hh_rlhf_dpo.csv")

print(f"Saving formatted CSV to: {output_file}")
df.to_csv(output_file, index=False, escapechar="\\")

print(f"Saved {len(df)} rows successfully!")
print(df.head())

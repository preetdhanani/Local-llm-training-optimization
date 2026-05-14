"""Dataset cleaning utilities for preference datasets."""
import re
import hashlib
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer


# Configuration
MAX_LEN_DELTA = 200      # Max token diff between chosen/rejected
MAX_RATIO = 2.5          # Max length ratio (longer / shorter)
MIN_TURN_TOKENS = 10     # Minimum tokens per turn

SYCOPHANCY_PHRASES = [
    r"you('re| are) (absolutely|totally|completely) right",
    r"great (question|point|observation)",
    r"i (completely|totally|fully) agree",
    r"you've (made|raised) (a )?(great|excellent|wonderful) point",
    r"that('s| is) (a )?(great|excellent|wonderful|fantastic)",
    r"absolutely[,!]",
    r"100% (correct|right|agree)",
]
SYCOPHANCY_RE = re.compile("|".join(SYCOPHANCY_PHRASES), re.IGNORECASE)


def parse_dialogue(text: str) -> list[dict]:
    """Parse dialogue into user/assistant turns."""
    parts = re.split(r"(?=\n\nHuman:|\n\nAssistant:)", text.strip())
    msgs = []
    for part in parts:
        part = part.strip()
        if part.startswith("Human:"):
            content = part[6:].strip()
            if content:
                msgs.append({"role": "user", "content": content})
        elif part.startswith("Assistant:"):
            content = part[10:].strip()
            if content:
                msgs.append({"role": "assistant", "content": content})
    return msgs


def get_last_assistant_turn(text: str) -> str:
    """Extract final assistant response."""
    turns = parse_dialogue(text)
    for t in reversed(turns):
        if t["role"] == "assistant":
            return t["content"]
    return ""


def make_fingerprint(chosen: str, rejected: str) -> str:
    """SHA256 hash for deduplication (catches swapped pairs)."""
    key = "|||".join(sorted([chosen.strip(), rejected.strip()]))
    return hashlib.sha256(key.encode()).hexdigest()


def remove_duplicates(dataset: Dataset) -> Dataset:
    """Remove exact and near-duplicate pairs."""
    seen = set()
    keep_flags = []
    for row in dataset:
        fp = make_fingerprint(row["chosen"], row["rejected"])
        if fp in seen:
            keep_flags.append(False)
        else:
            seen.add(fp)
            keep_flags.append(True)
    dataset = dataset.filter(lambda _, idx: keep_flags[idx], with_indices=True)
    print(f"[Dedup] kept {sum(keep_flags):,} / {len(keep_flags):,} rows")
    return dataset


def filter_length_bias(example: dict, tokenizer) -> bool:
    """Remove pairs with extreme length differences."""
    chosen_last = get_last_assistant_turn(example["chosen"])
    rejected_last = get_last_assistant_turn(example["rejected"])
    if not chosen_last or not rejected_last:
        return False

    c_len = len(tokenizer.encode(chosen_last, add_special_tokens=False))
    r_len = len(tokenizer.encode(rejected_last, add_special_tokens=False))

    # Check absolute delta
    if abs(c_len - r_len) > MAX_LEN_DELTA:
        return False

    # Check ratio
    longer, shorter = max(c_len, r_len), min(c_len, r_len)
    if shorter == 0 or (longer / shorter) > MAX_RATIO:
        return False

    return True


def sycophancy_score(text: str) -> int:
    """Count sycophancy patterns in text."""
    return len(SYCOPHANCY_RE.findall(text))


def filter_sycophancy(example: dict) -> bool:
    """Remove pairs where chosen is more sycophantic than rejected."""
    chosen_last = get_last_assistant_turn(example["chosen"])
    rejected_last = get_last_assistant_turn(example["rejected"])
    return sycophancy_score(chosen_last) <= sycophancy_score(rejected_last)


def repetition_ratio(text: str, ngram: int = 5) -> float:
    """Calculate n-gram repetition ratio."""
    words = text.lower().split()
    
    if len(words) <= ngram:
        return 0.0

    grams = [tuple(words[i:i+ngram]) for i in range(len(words) - ngram + 1)]
    
    if not grams:
        return 0.0

    unique = len(set(grams))
    return 1.0 - (unique / len(grams))


def filter_trivial_pairs(example: dict, tokenizer) -> bool:
    """Remove nearly-identical chosen/rejected pairs (Jaccard > 0.92)."""
    chosen_last = get_last_assistant_turn(example["chosen"])
    rejected_last = get_last_assistant_turn(example["rejected"])

    chosen_toks = set(tokenizer.encode(chosen_last, add_special_tokens=False))
    rejected_toks = set(tokenizer.encode(rejected_last, add_special_tokens=False))

    if not chosen_toks or not rejected_toks:
        return False

    intersection = chosen_toks & rejected_toks
    union = chosen_toks | rejected_toks
    jaccard = len(intersection) / len(union)

    return jaccard < 0.92


def filter_minimum_length(example: dict, tokenizer) -> bool:
    """Require minimum tokens in both responses."""
    c = get_last_assistant_turn(example["chosen"])
    r = get_last_assistant_turn(example["rejected"])
    c_len = len(tokenizer.encode(c, add_special_tokens=False))
    r_len = len(tokenizer.encode(r, add_special_tokens=False))
    return c_len >= MIN_TURN_TOKENS and r_len >= MIN_TURN_TOKENS


def log_balance_stats(dataset: Dataset, tokenizer, n_sample: int = 2000) -> None:
    """Print length distribution statistics."""
    sample = dataset.select(range(min(n_sample, len(dataset))))

    chosen_lens, rejected_lens, deltas = [], [], []
    for row in sample:
        c = get_last_assistant_turn(row["chosen"])
        r = get_last_assistant_turn(row["rejected"])
        cl = len(tokenizer.encode(c, add_special_tokens=False))
        rl = len(tokenizer.encode(r, add_special_tokens=False))
        chosen_lens.append(cl)
        rejected_lens.append(rl)
        deltas.append(abs(cl - rl))

    print("\n── Dataset Balance Stats (sample) ──────────────────────")
    print(f"  chosen   len  mean={np.mean(chosen_lens):.0f}  "
          f"std={np.std(chosen_lens):.0f}  "
          f"p95={np.percentile(chosen_lens, 95):.0f}")
    print(f"  rejected len  mean={np.mean(rejected_lens):.0f}  "
          f"std={np.std(rejected_lens):.0f}  "
          f"p95={np.percentile(rejected_lens, 95):.0f}")
    print(f"  |delta|       mean={np.mean(deltas):.0f}  "
          f"max={np.max(deltas):.0f}  "
          f"p95={np.percentile(deltas, 95):.0f}")
    print("────────────────────────────────────────────────────────\n")


def validate_dpo_row(row: dict) -> tuple[bool, str]:
    """
    Validate a single DPO row has all required fields with valid content.
    
    Args:
        row: Dictionary with "prompt", "chosen", "rejected" fields
        
    Returns:
        Tuple of (is_valid, reason_string)
    """
    for field in ["prompt", "chosen", "rejected"]:
        val = row.get(field)
        # Check for None
        if val is None:
            return False, f"{field} is None"
        # Check for non-string types (shouldn't happen but be safe)
        if not isinstance(val, str):
            return False, f"{field} is {type(val).__name__}, not str"
        # Check for empty strings
        if len(val.strip()) == 0:
            return False, f"{field} is empty string"
    return True, "valid"


def validate_dpo_dataset(
    dataset: Dataset,
    dataset_name: str = "dataset",
    max_errors_shown: int = 5,
) -> None:
    """
    Validate an entire DPO dataset for problematic rows.
    
    Args:
        dataset: Dataset to validate
        dataset_name: Name for error reporting (e.g., "train", "eval")
        max_errors_shown: Max number of errors to display
        
    Raises:
        ValueError: If any invalid rows are found
    """
    import sys
    
    invalid_rows = []
    for i, row in enumerate(dataset):
        valid, reason = validate_dpo_row(row)
        if not valid:
            invalid_rows.append((i, reason))
    
    if invalid_rows:
        sys.stderr.write(f"[ERROR] {dataset_name} dataset has {len(invalid_rows)} invalid rows:\n")
        for idx, reason in invalid_rows[:max_errors_shown]:
            sys.stderr.write(f"  Row {idx}: {reason}\n")
        sys.stderr.flush()
        raise ValueError(f"{dataset_name} dataset validation failed: {len(invalid_rows)} invalid rows")
    
    sys.stderr.write(f"[DEBUG DPO] {dataset_name} validation OK: {len(dataset)} rows checked\n")
    sys.stderr.flush()


def clean_preference_dataset(
    dataset: Dataset,
    tokenizer,
    verbose: bool = True,
) -> Dataset:
    """Master cleaning pipeline."""
    
    def log(tag: str, before: int, after: int):
        dropped = before - after
        pct = (dropped / before * 100) if before else 0
        print(f"[{tag:<22}]  {after:>7,}  (dropped {dropped:>6,}  {pct:.1f}%)")

    n0 = len(dataset)
    print(f"\n{'─'*55}")
    print(f"  Starting rows: {n0:,}")
    print(f"{'─'*55}")

    # 1. Exact/near duplicates
    dataset = remove_duplicates(dataset)
    log("exact dedup", n0, len(dataset)); n1 = len(dataset)

    # 2. Minimum length
    dataset = dataset.filter(
        lambda x: filter_minimum_length(x, tokenizer), num_proc=1
    )
    log("min length", n1, len(dataset)); n2 = len(dataset)

    # 3. Length bias
    dataset = dataset.filter(
        lambda x: filter_length_bias(x, tokenizer), num_proc=1
    )
    log("length bias", n2, len(dataset)); n3 = len(dataset)

    # 4. Sycophancy bias
    dataset = dataset.filter(filter_sycophancy, num_proc=1)
    log("sycophancy bias", n3, len(dataset)); n4 = len(dataset)

    # 5. Trivial/near-identical pairs
    dataset = dataset.filter(
        lambda x: filter_trivial_pairs(x, tokenizer), num_proc=1
    )
    log("trivial pairs", n4, len(dataset)); n5 = len(dataset)

    print(f"{'─'*55}")
    print(f"  Final rows : {n5:,}  ({n5/n0*100:.1f}% of original)")
    print(f"{'─'*55}\n")

    if verbose:
        log_balance_stats(dataset, tokenizer)

    return dataset

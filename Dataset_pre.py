# ─────────────────────────────────────────────────────────────
#  dataset_cleaner.py
#  Cleans hh-rlhf (or any preference dataset) before RM training
#  Targets: length bias, sycophancy bias, RM overfitting
# ─────────────────────────────────────────────────────────────
import re
import hashlib
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from collections import defaultdict

TOKENIZER_ID     = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LEN          = 512
MIN_TURN_TOKENS  = 10

# ── Length bias thresholds ────────────────────────────────────
MAX_LEN_DELTA    = 200    # filter if |chosen_len - rejected_len| > this
MAX_RATIO        = 2.5    # filter if longer/shorter > this ratio

# ── Sycophancy keywords ───────────────────────────────────────
SYCOPHANCY_PHRASES = [
    r"you('re| are) (absolutely|totally|completely) right",
    r"great (question|point|observation)",
    r"i (completely|totally|fully) agree",
    r"you've (made|raised) (a )?(great|excellent|wonderful) point",
    r"that('s| is) (a )?(great|excellent|wonderful|fantastic)",
    r"absolutely[,!]",
    r"100% (correct|right|agree)",
]
SYCOPHANCY_RE = re.compile(
    "|".join(SYCOPHANCY_PHRASES),
    re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────
#  Dialogue Parser (robust)
# ─────────────────────────────────────────────────────────────
def parse_dialogue(text: str) -> list[dict]:
    parts = re.split(r"(?=\n\nHuman:|\n\nAssistant:)", text.strip())
    msgs  = []
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
    """Extract only the final assistant response — what the RM actually scores."""
    turns = parse_dialogue(text)
    for t in reversed(turns):
        if t["role"] == "assistant":
            return t["content"]
    return ""


def get_all_user_turns(text: str) -> list[str]:
    return [t["content"] for t in parse_dialogue(text) if t["role"] == "user"]


# ─────────────────────────────────────────────────────────────
#  Step 1 — Exact & Near-Duplicate Removal
# ─────────────────────────────────────────────────────────────
def make_fingerprint(chosen: str, rejected: str) -> str:
    """
    SHA256 of sorted(chosen, rejected) — catches swapped duplicates too.
    """
    key = "|||".join(sorted([chosen.strip(), rejected.strip()]))
    return hashlib.sha256(key.encode()).hexdigest()


def remove_duplicates(dataset: Dataset) -> Dataset:
    seen       = set()
    keep_flags = []

    for row in dataset:
        fp = make_fingerprint(row["chosen"], row["rejected"])
        if fp in seen:
            keep_flags.append(False)
        else:
            seen.add(fp)
            keep_flags.append(True)

    dataset = dataset.filter(lambda _, idx: keep_flags[idx], with_indices=True)
    print(f"[Dedup]  kept {sum(keep_flags):,} / {len(keep_flags):,} rows")
    return dataset


# ─────────────────────────────────────────────────────────────
#  Step 2 — Length Bias Filters
# ─────────────────────────────────────────────────────────────
def filter_length_bias(example: dict, tokenizer) -> bool:
    """
    Two checks:
      1. Absolute delta  — |chosen_len - rejected_len| <= MAX_LEN_DELTA
      2. Ratio check     — longer / shorter <= MAX_RATIO

    Both use token counts, not character counts (more accurate).
    """
    chosen_last   = get_last_assistant_turn(example["chosen"])
    rejected_last = get_last_assistant_turn(example["rejected"])

    if not chosen_last or not rejected_last:
        return False

    c_len = len(tokenizer.encode(chosen_last, add_special_tokens=False))
    r_len = len(tokenizer.encode(rejected_last, add_special_tokens=False))

    # absolute delta
    if abs(c_len - r_len) > MAX_LEN_DELTA:
        return False

    # ratio check — prevents 5-token vs 300-token comparisons
    longer, shorter = max(c_len, r_len), min(c_len, r_len)
    if shorter == 0 or (longer / shorter) > MAX_RATIO:
        return False

    return True


# ─────────────────────────────────────────────────────────────
#  Step 3 — Sycophancy Bias Filters
# ─────────────────────────────────────────────────────────────
def sycophancy_score(text: str) -> int:
    """Count how many sycophancy patterns appear in text."""
    return len(SYCOPHANCY_RE.findall(text))


def filter_sycophancy(example: dict) -> bool:
    """
    Remove pairs where CHOSEN is more sycophantic than REJECTED.
    These pairs teach the RM to prefer agreement over correctness.

    Keep pairs where:
      - Neither side is sycophantic (clean signal)
      - Rejected is more sycophantic (correct label direction)
    """
    chosen_last   = get_last_assistant_turn(example["chosen"])
    rejected_last = get_last_assistant_turn(example["rejected"])

    chosen_score   = sycophancy_score(chosen_last)
    rejected_score = sycophancy_score(rejected_last)

    # bad: chosen is MORE sycophantic → wrong signal
    if chosen_score > rejected_score:
        return False

    return True


# ─────────────────────────────────────────────────────────────
#  Step 4 — Reward Hacking Prevention Filters
# ─────────────────────────────────────────────────────────────

# Formatting tricks the RM learns to blindly reward
FORMATTING_TRICK_RE = re.compile(
    r"(#{1,6}\s.+\n.*){4,}"       # 4+ markdown headers back to back
    r"|(\*{1,2}.+\*{1,2}\s*){6,}" # 6+ bold/italic fragments
    r"|(\n\s*[-*]\s.+){8,}",       # 8+ consecutive bullet points
    re.DOTALL
)


def repetition_ratio(text: str, ngram: int = 5) -> float:
    words = text.lower().split()
    if len(words) < ngram:       # ← catches < but misses the == case
        return 0.0
    grams  = [tuple(words[i:i+ngram]) for i in range(len(words) - ngram)]
    # if len(words) == ngram → range(0) → grams = [] → CRASH
    unique = len(set(grams))
    return 1.0 - (unique / len(grams))



def repetition_ratio(text: str, ngram: int = 5) -> float:
    words = text.lower().split()
    
    if len(words) <= ngram:   # FIX: <= instead of <
        return 0.0

    grams = [tuple(words[i:i+ngram]) for i in range(len(words) - ngram + 1)]
    
    if not grams:             # defensive guard (extra safety)
        return 0.0

    unique = len(set(grams))
    return 1.0 - (unique / len(grams))


# ─────────────────────────────────────────────────────────────
#  Step 5 — RM Overfitting Guards
# ─────────────────────────────────────────────────────────────
def filter_trivial_pairs(example: dict, tokenizer) -> bool:
    """
    Remove pairs that are TOO easy — near-identical chosen/rejected.
    RM memorises these instead of learning signal.

    Uses token overlap (Jaccard similarity):
      similarity > 0.92 → trivial pair → discard
    """
    chosen_last   = get_last_assistant_turn(example["chosen"])
    rejected_last = get_last_assistant_turn(example["rejected"])

    chosen_toks   = set(tokenizer.encode(chosen_last,   add_special_tokens=False))
    rejected_toks = set(tokenizer.encode(rejected_last, add_special_tokens=False))

    if not chosen_toks or not rejected_toks:
        return False

    intersection = chosen_toks & rejected_toks
    union        = chosen_toks | rejected_toks
    jaccard      = len(intersection) / len(union)

    return jaccard < 0.92   # keep pairs that are meaningfully different


def filter_minimum_length(example: dict, tokenizer) -> bool:
    """Both sides must have at least MIN_TURN_TOKENS — removes empty/broken rows."""
    c = get_last_assistant_turn(example["chosen"])
    r = get_last_assistant_turn(example["rejected"])
    c_len = len(tokenizer.encode(c, add_special_tokens=False))
    r_len = len(tokenizer.encode(r, add_special_tokens=False))
    return c_len >= MIN_TURN_TOKENS and r_len >= MIN_TURN_TOKENS


# ─────────────────────────────────────────────────────────────
#  Step 6 — Dataset Balance Check
# ─────────────────────────────────────────────────────────────
def log_balance_stats(dataset: Dataset, tokenizer, n_sample: int = 2000) -> None:
    """
    Print length distribution stats on a sample.
    Large mean delta → residual length bias remaining after filters.
    """
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
    print(f"  target: mean delta << {MAX_LEN_DELTA}, p95 << {MAX_LEN_DELTA}")
    print("────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────
#  Master Cleaning Pipeline
# ─────────────────────────────────────────────────────────────
def clean_preference_dataset(
    dataset:    Dataset,
    tokenizer,
    verbose:    bool = True,
) -> Dataset:

    def log(tag: str, before: int, after: int):
        dropped = before - after
        pct     = (dropped / before * 100) if before else 0
        print(f"[{tag:<22}]  {after:>7,}  (dropped {dropped:>6,}  {pct:.1f}%)")

    n0 = len(dataset)
    print(f"\n{'─'*55}")
    print(f"  Starting rows: {n0:,}")
    print(f"{'─'*55}")

    # ── 1. exact/near duplicates ──────────────────────────────
    dataset = remove_duplicates(dataset)
    log("exact dedup", n0, len(dataset)); n1 = len(dataset)

    # ── 2. minimum length ─────────────────────────────────────
    dataset = dataset.filter(
        lambda x: filter_minimum_length(x, tokenizer), num_proc=4
    )
    log("min length", n1, len(dataset)); n2 = len(dataset)

    # ── 3. length bias ────────────────────────────────────────
    dataset = dataset.filter(
        lambda x: filter_length_bias(x, tokenizer), num_proc=4
    )
    log("length bias", n2, len(dataset)); n3 = len(dataset)

    # ── 4. sycophancy bias ────────────────────────────────────
    dataset = dataset.filter(filter_sycophancy, num_proc=4)
    log("sycophancy bias", n3, len(dataset)); n4 = len(dataset)

 
    # ── 6. trivial/near-identical pairs ──────────────────────
    dataset = dataset.filter(
        lambda x: filter_trivial_pairs(x, tokenizer), num_proc=4
    )
    log("trivial pairs", n4, len(dataset)); n6 = len(dataset)

    print(f"{'─'*55}")
    print(f"  Final rows : {n6:,}  ({n6/n0*100:.1f}% of original)")
    print(f"{'─'*55}\n")

    if verbose:
        log_balance_stats(dataset, tokenizer)

    return dataset


# ─────────────────────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw = load_dataset("Anthropic/hh-rlhf", split="train")

    cleaned = clean_preference_dataset(raw, tokenizer, verbose=True)

    # save for reuse — don't rerun cleaning every training run
    cleaned.save_to_disk("./hh_rlhf_cleaned")
    print("Saved to ./hh_rlhf_cleaned")
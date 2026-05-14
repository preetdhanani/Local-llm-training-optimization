"""Data filtering and validation functions."""
from typing import Callable, Dict

from datasets import Dataset


def validate_pair(
    prompt: str,
    chosen: str,
    rejected: str,
    min_prompt_len: int = 10,
    min_response_len: int = 3,
    max_response_len: int = 300,
) -> bool:
    """
    Validate a preference pair (prompt, chosen, rejected).

    Args:
        prompt: User prompt
        chosen: Preferred response
        rejected: Rejected response
        min_prompt_len: Minimum prompt length
        min_response_len: Minimum response length
        max_response_len: Maximum response length

    Returns:
        True if pair is valid, False otherwise
    """
    if not prompt or not chosen or not rejected:
        return False

    if len(prompt) < min_prompt_len:
        return False

    if len(chosen) < min_response_len or len(rejected) < min_response_len:
        return False

    if len(chosen) > max_response_len or len(rejected) > max_response_len:
        return False

    return True


def validate_pair_by_tokens(
    prompt: str,
    chosen: str,
    rejected: str,
    tokenizer,
    max_prompt_tokens: int = 256,
    max_response_tokens: int = 512,
    max_length_diff: int = 100,
) -> bool:
    """
    Validate a pair at token level (catches truncation bias).

    Args:
        prompt: User prompt
        chosen: Preferred response
        rejected: Rejected response
        tokenizer: Tokenizer for token counting
        max_prompt_tokens: Max tokens for prompt
        max_response_tokens: Max tokens for response
        max_length_diff: Max token diff between chosen and rejected

    Returns:
        True if pair passes token validation
    """
    p_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    c_tokens = len(tokenizer(chosen, add_special_tokens=False)["input_ids"])
    r_tokens = len(tokenizer(rejected, add_special_tokens=False)["input_ids"])

    if p_tokens > max_prompt_tokens:
        return False

    if c_tokens > max_response_tokens or r_tokens > max_response_tokens:
        return False

    if abs(c_tokens - r_tokens) > max_length_diff:
        return False

    return True


def filter_by_length(
    example: dict,
    tokenizer,
    limits: dict,
) -> bool:
    """
    Filter example based on character and token length constraints.

    Args:
        example: Dataset example
        tokenizer: Tokenizer
        limits: Dict with keys: min_prompt_len, min_response_len, max_response_len,
                max_prompt_tokens, max_response_tokens, max_length_diff

    Returns:
        True if example passes filtering
    """
    if not example.get("valid"):
        return False

    p = example.get("prompt", "")
    c = example.get("chosen", "")
    r = example.get("rejected", "")

    # Character-level validation
    char_valid = validate_pair(
        p, c, r,
        min_prompt_len=limits.get("min_prompt_len", 10),
        min_response_len=limits.get("min_response_len", 3),
        max_response_len=limits.get("max_response_len", 300),
    )
    if not char_valid:
        return False

    # Token-level validation
    token_valid = validate_pair_by_tokens(
        p, c, r,
        tokenizer,
        max_prompt_tokens=limits.get("max_prompt_tokens", 256),
        max_response_tokens=limits.get("max_response_tokens", 512),
        max_length_diff=limits.get("max_length_diff", 100),
    )
    return token_valid


def remove_duplicates(dataset: Dataset) -> Dataset:
    """
    Remove exact and near-duplicate examples.

    Args:
        dataset: HuggingFace Dataset

    Returns:
        Filtered dataset
    """
    import hashlib

    seen = set()
    keep_flags = []

    for row in dataset:
        chosen = row.get("chosen", "").strip()
        rejected = row.get("rejected", "").strip()
        # Fingerprint: hash of sorted pair (catches swapped duplicates)
        key = "|||".join(sorted([chosen, rejected]))
        fp = hashlib.sha256(key.encode()).hexdigest()

        if fp in seen:
            keep_flags.append(False)
        else:
            seen.add(fp)
            keep_flags.append(True)

    filtered = dataset.filter(
        lambda _, idx: keep_flags[idx],
        with_indices=True,
    )
    return filtered


def dataset_stats(dataset: Dataset) -> dict:
    """
    Compute and return dataset statistics.

    Args:
        dataset: HuggingFace Dataset with "chosen" and "rejected" fields

    Returns:
        Dict of statistics
    """
    if len(dataset) == 0:
        return {}

    lengths_c = [len(x.get("chosen", "")) for x in dataset]
    lengths_r = [len(x.get("rejected", "")) for x in dataset]
    margins = [
        len(x.get("chosen", "")) - len(x.get("rejected", ""))
        for x in dataset
    ]

    stats = {
        "num_examples": len(dataset),
        "avg_chosen_length": sum(lengths_c) / len(lengths_c) if lengths_c else 0,
        "avg_rejected_length": sum(lengths_r) / len(lengths_r) if lengths_r else 0,
        "avg_length_diff": sum(abs(m) for m in margins) / len(margins) if margins else 0,
        "max_chosen_length": max(lengths_c) if lengths_c else 0,
        "min_chosen_length": min(lengths_c) if lengths_c else 0,
    }

    return stats

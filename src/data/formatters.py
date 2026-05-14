"""Data formatting functions for SFT and DPO."""
from typing import Optional


def safe_split(text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Safely split text into prompt and response by "Assistant:" delimiter.

    Args:
        text: Raw conversation text

    Returns:
        Tuple of (prompt, response) or (None, None) if parsing fails
    """
    if not isinstance(text, str):
        return None, None

    parts = text.rsplit("Assistant:", 1)
    if len(parts) != 2:
        return None, None

    prompt = parts[0].strip()
    response = parts[1].strip()
    return prompt, response


def parse_dialogue(text: str) -> list[dict]:
    """
    Parse dialogue text into a list of message dicts.

    Args:
        text: Raw conversation text with "Human:" and "Assistant:" delimiters

    Returns:
        List of {"role": "user"|"assistant", "content": "..."} dicts
    """
    import re

    parts = re.split(r"(?=\n\nHuman:|\n\nAssistant:)", text.strip())
    messages = []

    for part in parts:
        part = part.strip()
        if part.startswith("Human:"):
            content = part[6:].strip()
            if content:
                messages.append({"role": "user", "content": content})
        elif part.startswith("Assistant:"):
            content = part[10:].strip()
            if content:
                messages.append({"role": "assistant", "content": content})

    return messages


def format_for_sft(example: dict, tokenizer) -> dict:
    """
    Format raw dataset example for SFT training.

    Converts chosen text into chat template format.

    Args:
        example: Dataset row with "chosen" field
        tokenizer: Tokenizer with apply_chat_template method

    Returns:
        Dict with "text" field containing formatted conversation
    """
    try:
        text = example.get("chosen", "")
        
        # Handle missing or non-string data
        if not text or not isinstance(text, str):
            return {"text": ""}
        
        # Use raw text directly - avoid apply_chat_template to prevent hangs
        return {"text": text.strip()}
        
    except Exception as e:
        import sys
        print(f"[WARN] format_for_sft error: {e}", file=sys.stderr, flush=True)
        return {"text": ""}


def format_preference_pair(example: dict) -> dict:
    """
    Format raw dataset example for DPO training.

    Parses chosen/rejected pairs and validates them.

    Args:
        example: Dataset row with "chosen" and "rejected" fields

    Returns:
        Dict with "prompt", "chosen", "rejected", "valid" fields
    """
    chosen_prompt, chosen_response = safe_split(example.get("chosen"))
    rejected_prompt, rejected_response = safe_split(example.get("rejected"))

    valid = True

    # Both must have same prompt
    if chosen_prompt is None or rejected_prompt is None or chosen_prompt != rejected_prompt:
        valid = False

    # Both must have responses
    if chosen_response is None or rejected_response is None:
        valid = False

    return {
        "prompt": chosen_prompt if valid else "",
        "chosen": chosen_response if valid else "",
        "rejected": rejected_response if valid else "",
        "valid": valid,
    }

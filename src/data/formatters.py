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


def format_for_sft(example: dict, tokenizer, cfg) -> dict:
    """
    Format raw dataset example for SFT training.
    Supports standard messages or custom columns.
    """
    try:
        messages = []
        if cfg.dataset_format == "standard_messages":
            messages = example.get("messages", [])
        else:
            # custom_columns: build message list from mapped columns
            prompt = example.get(cfg.prompt_column, "")
            response = example.get(cfg.chosen_column, "")
            
            if cfg.system_prompt:
                messages.append({"role": "system", "content": cfg.system_prompt})
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": response})

        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}
        
    except Exception as e:
        import sys
        print(f"[WARN] format_for_sft error: {e}", file=sys.stderr, flush=True)
        return {"text": ""}


def format_preference_pair(example: dict, cfg) -> dict:
    """
    Format raw dataset example for DPO training.
    Supports standard messages or custom columns.
    """
    if cfg.dataset_format == "standard_messages":
        prompt = example.get("prompt", "")
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")
    else:
        # custom_columns mapping
        prompt = example.get(cfg.prompt_column, "")
        chosen = example.get(cfg.chosen_column, "")
        rejected = example.get(cfg.rejected_column, "")

    valid = bool(prompt and chosen and rejected)
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "valid": valid,
    }

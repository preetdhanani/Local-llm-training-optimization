"""Path utilities for managing timestamped output directories."""
import os
from datetime import datetime


def get_timestamped_output_dir(base_dir: str, prefix: str = "") -> str:
    """
    Generate a timestamped output directory path.
    
    Format: {base_dir}/{prefix}_output_{YYYY-MM-DD}_{HH-MM-SS}
    Example: ./outputs/sft_output_2026-05-11_18-35-42
    
    Args:
        base_dir: Base directory (e.g., "./outputs")
        prefix: Prefix for directory name (e.g., "sft", "dpo")
    
    Returns:
        Full path to timestamped directory (not created yet)
    """
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if prefix:
        dir_name = f"{prefix}_output_{timestamp}"
    else:
        dir_name = f"output_{timestamp}"
    
    return os.path.join(base_dir, dir_name)


def find_latest_sft_output(base_dir: str = "./outputs") -> str:
    """
    Find the most recently created SFT output directory.
    
    Looks for directories matching pattern: sft_output_YYYY-MM-DD_HH-MM-SS
    
    Args:
        base_dir: Base directory containing timestamped outputs
    
    Returns:
        Path to latest SFT output directory
        
    Raises:
        FileNotFoundError: If no SFT output directory found
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Output directory not found: {base_dir}")
    
    sft_dirs = [
        d for d in os.listdir(base_dir)
        if d.startswith("sft_output_")
    ]
    
    if not sft_dirs:
        raise FileNotFoundError(f"No SFT output directories found in: {base_dir}")
    
    # Sort by name (timestamp format ensures lexicographic ordering = chronological)
    latest = sorted(sft_dirs)[-1]
    latest_path = os.path.join(base_dir, latest)
    
    return latest_path

"""Logger setup utility."""
import logging
import os
from datetime import datetime
from typing import Optional


def get_timestamped_log_path(trainer_name: str) -> str:
    """
    Generate a timestamped log file path.
    
    Format: logs/{trainer_name}_train_{YYYY-MM-DD}_{HH-MM-SS}.log
    Example: logs/sft_train_2026-05-11_14-35-42.log
    
    Args:
        trainer_name: Name of trainer (e.g., "sft", "dpo")
    
    Returns:
        Full path to timestamped log file
    """
    try:
        os.makedirs("logs", exist_ok=True)
    except FileExistsError:
        if not os.path.isdir("logs"):
            raise
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"logs/{trainer_name}_train_{timestamp}.log"


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name
        log_file: Path to log file (if None, console only)
        level: Logging level (INFO, DEBUG, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (always)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else "."
        try:
            os.makedirs(log_dir, exist_ok=True)
        except FileExistsError:
            if not os.path.isdir(log_dir):
                raise
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.flush()
        logger.addHandler(file_handler)

    return logger

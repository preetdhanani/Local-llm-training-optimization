"""Weights & Biases helper utilities."""
from dataclasses import asdict
from typing import Any, Dict, Iterable, Optional
import os
import re

from src.config import TrainConfig


def _safe_import_wandb():
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    return wandb


def _parse_tags(tags: Optional[str]) -> Optional[Iterable[str]]:
    if not tags:
        return None
    return [t.strip() for t in tags.split(",") if t.strip()]


def _sanitize_artifact_name(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("=", "_")
    )


def _sanitize_name_token(value: str) -> str:
    value = value.replace("/", "-")
    value = re.sub(r"[^A-Za-z0-9_.-]", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "unknown"


def format_run_name(stage: str, model_id: str, timestamp: str) -> str:
    safe_stage = _sanitize_name_token(stage)
    safe_model = _sanitize_name_token(model_id)
    safe_ts = _sanitize_name_token(timestamp)
    return f"{safe_stage}-{safe_model}-{safe_ts}"


def format_group_name(model_id: str, timestamp: str) -> str:
    safe_model = _sanitize_name_token(model_id)
    safe_ts = _sanitize_name_token(timestamp)
    return f"rlhf-{safe_model}-{safe_ts}"


def should_enable_wandb(cfg: TrainConfig) -> bool:
    return bool(cfg.wandb_project) and cfg.wandb_mode != "disabled"


def init_wandb_run(
    cfg: TrainConfig,
    run_name: str,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
):
    """Initialize a W&B run if enabled; returns run or None."""
    if not should_enable_wandb(cfg):
        return None

    wandb = _safe_import_wandb()
    if wandb is None:
        return None

    config = asdict(cfg)
    if config_overrides:
        config.update(config_overrides)

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        group=group,
        job_type=job_type,
        tags=_parse_tags(cfg.wandb_tags),
        notes=cfg.wandb_notes,
        config=config,
        mode=cfg.wandb_mode,
    )

    return run


def log_metrics(cfg: TrainConfig, metrics: Dict[str, Any]) -> None:
    if not should_enable_wandb(cfg):
        return
    wandb = _safe_import_wandb()
    if wandb is None or wandb.run is None:
        return
    wandb.log(metrics)


def log_dataset_stats(cfg: TrainConfig, prefix: str, stats: Dict[str, Any]) -> None:
    if not cfg.wandb_log_dataset_stats:
        return
    log_metrics(cfg, {f"{prefix}/{key}": value for key, value in stats.items()})


def log_table(cfg: TrainConfig, name: str, columns: Iterable[str], rows: Iterable[Iterable[Any]]) -> None:
    if not cfg.wandb_log_samples:
        return
    if not should_enable_wandb(cfg):
        return
    wandb = _safe_import_wandb()
    if wandb is None or wandb.run is None:
        return
    table = wandb.Table(columns=list(columns))
    for row in rows:
        table.add_data(*row)
    wandb.log({name: table})


def log_dir_artifact(
    cfg: TrainConfig,
    dir_path: str,
    name: str,
    artifact_type: str = "model",
    aliases: Optional[Iterable[str]] = None,
) -> None:
    if not should_enable_wandb(cfg) or not cfg.wandb_log_artifacts:
        return
    wandb = _safe_import_wandb()
    if wandb is None or wandb.run is None:
        return
    if not os.path.isdir(dir_path):
        return

    artifact = wandb.Artifact(_sanitize_artifact_name(name), type=artifact_type)
    artifact.add_dir(dir_path)
    wandb.log_artifact(artifact, aliases=list(aliases) if aliases else None)


def log_file_artifact(
    cfg: TrainConfig,
    file_path: str,
    name: str,
    artifact_type: str = "logs",
    aliases: Optional[Iterable[str]] = None,
) -> None:
    if not should_enable_wandb(cfg) or not cfg.wandb_log_artifacts:
        return
    wandb = _safe_import_wandb()
    if wandb is None or wandb.run is None:
        return
    if not os.path.isfile(file_path):
        return

    artifact = wandb.Artifact(_sanitize_artifact_name(name), type=artifact_type)
    artifact.add_file(file_path)
    wandb.log_artifact(artifact, aliases=list(aliases) if aliases else None)


def finish_run(cfg: TrainConfig) -> None:
    if not should_enable_wandb(cfg):
        return
    wandb = _safe_import_wandb()
    if wandb is None or wandb.run is None:
        return
    wandb.finish()

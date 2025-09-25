"""Utility helpers shared across sentiment-analysis modules."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

RESULTS_DIR: Path = Path("results")
ARTIFACTS_DIR: Path = Path("artifacts")


def ensure_dir(path: Path | str) -> Path:
    """Create *path* (and parents) if it does not exist and return it."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    """Seed *random*, *numpy*, and *torch* for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - optional GPU
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():  # pragma: no cover - Apple Silicon
        torch.mps.manual_seed(seed)


def _json_default(obj: Any) -> Any:
    """Best-effort JSON serialiser for dataclasses and numpy scalars."""

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def save_json(path: Path | str, payload: Dict[str, Any]) -> Path:
    """Save *payload* to *path* (creating parent folders) and return the path."""

    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_json_default)
        fh.write("\n")
    return path


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool *hidden_states* using *attention_mask* to ignore padding."""

    mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


__all__ = [
    "RESULTS_DIR",
    "ARTIFACTS_DIR",
    "ensure_dir",
    "set_seed",
    "save_json",
    "masked_mean",
]

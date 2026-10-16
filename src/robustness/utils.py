"""Small reproducibility, device, and artifact helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without promising cross-platform identity."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Use CUDA only when PyTorch reports that it is usable."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_experiment(config: ExperimentConfig) -> tuple[torch.device, Path]:
    """Validate settings, seed libraries, and create the writable output directory."""
    config.validate()
    seed_everything(config.seed)
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return select_device(), output_dir


def save_json_atomic(payload: Any, destination: str | Path) -> Path:
    """Write JSON through a temporary sibling so partial files are not retained."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)
    return path


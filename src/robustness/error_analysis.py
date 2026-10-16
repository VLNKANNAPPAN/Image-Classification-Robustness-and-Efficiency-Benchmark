"""Visual inspection helpers for high-confidence and representative mistakes."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class Misclassification:
    sample_index: int
    true_index: int
    predicted_index: int
    confidence: float
    image: torch.Tensor


@torch.inference_mode()
def collect_misclassifications(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit: int = 12,
) -> list[Misclassification]:
    """Collect the highest-confidence wrong predictions from an ordered loader."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    model.eval()
    errors: list[Misclassification] = []
    sample_offset = 0
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        probabilities = logits.softmax(dim=1)
        confidence, predictions = probabilities.max(dim=1)
        wrong_positions = (predictions.cpu() != labels).nonzero(as_tuple=False).flatten()
        for position in wrong_positions.tolist():
            errors.append(
                Misclassification(
                    sample_index=sample_offset + position,
                    true_index=int(labels[position]),
                    predicted_index=int(predictions[position].cpu()),
                    confidence=float(confidence[position].cpu()),
                    image=images[position].detach().cpu().clone(),
                )
            )
        sample_offset += labels.size(0)
    return sorted(errors, key=lambda item: item.confidence, reverse=True)[:limit]


def _display_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def save_error_gallery(
    errors: Sequence[Misclassification],
    class_names: Sequence[str],
    model_name: str,
    condition: str,
    output_dir: str | Path,
) -> tuple[Path | None, Path]:
    """Save a compact figure plus a machine-readable manifest of observed mistakes."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "misclassifications.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "true_label", "predicted_label", "confidence", "model", "condition"])
        for error in errors:
            writer.writerow(
                [
                    error.sample_index,
                    class_names[error.true_index],
                    class_names[error.predicted_index],
                    error.confidence,
                    model_name,
                    condition,
                ]
            )
    if not errors:
        return None, manifest_path
    columns = min(4, len(errors))
    rows = (len(errors) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.6 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, error in zip(axes.flat, errors):
        axis.imshow(_display_image(error.image))
        axis.set_title(
            f"True: {class_names[error.true_index]}\nPred: {class_names[error.predicted_index]} ({error.confidence:.1%})"
        )
        axis.axis("off")
    figure.suptitle(f"{model_name} — {condition}: high-confidence errors")
    figure.tight_layout()
    gallery_path = root / "misclassifications.png"
    figure.savefig(gallery_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return gallery_path, manifest_path


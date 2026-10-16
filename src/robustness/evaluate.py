"""Reusable classification evaluation, persistence, and comparison figures."""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

from .utils import save_json_atomic


@dataclass(frozen=True)
class EvaluationResult:
    model: str
    condition: str
    sample_count: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    metrics_path: Path
    predictions_path: Path
    confusion_matrix_path: Path


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


@torch.inference_mode()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, predicted indices, and maximum softmax confidence."""
    model.eval()
    true_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probabilities = logits.softmax(dim=1)
        confidence, predictions = probabilities.max(dim=1)
        true_parts.append(labels.detach().cpu().numpy())
        predicted_parts.append(predictions.detach().cpu().numpy())
        confidence_parts.append(confidence.detach().cpu().numpy())
    if not true_parts:
        raise ValueError("Evaluation loader contains no samples")
    return (
        np.concatenate(true_parts),
        np.concatenate(predicted_parts),
        np.concatenate(confidence_parts),
    )


def calculate_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]
) -> dict:
    """Calculate stable overall and per-class metrics with explicit zero handling."""
    labels = list(range(len(class_names)))
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same non-zero length")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "sample_count": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class": {name: report[name] for name in class_names},
        "confusion_matrix": matrix.tolist(),
    }


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    destination: str | Path,
    title: str,
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    class_names: Sequence[str],
    device: torch.device,
    model_name: str,
    condition: str,
    output_dir: str | Path,
) -> EvaluationResult:
    """Evaluate one model/condition and save JSON, CSV predictions, and a matrix figure."""
    y_true, y_pred, confidence = collect_predictions(model, loader, device)
    metrics = calculate_classification_metrics(y_true, y_pred, class_names)
    metrics.update({"model": model_name, "condition": condition, "classes": list(class_names)})
    root = Path(output_dir) / _safe_name(model_name) / _safe_name(condition)
    metrics_path = save_json_atomic(metrics, root / "metrics.json")
    predictions_path = root / "predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "true_index", "true_label", "predicted_index", "predicted_label", "confidence"])
        for index, (truth, prediction, score) in enumerate(zip(y_true, y_pred, confidence)):
            writer.writerow(
                [index, int(truth), class_names[int(truth)], int(prediction), class_names[int(prediction)], float(score)]
            )
    matrix_path = save_confusion_matrix(
        np.asarray(metrics["confusion_matrix"]),
        class_names,
        root / "confusion_matrix.png",
        f"{model_name} — {condition}",
    )
    return EvaluationResult(
        model=model_name,
        condition=condition,
        sample_count=metrics["sample_count"],
        accuracy=metrics["accuracy"],
        macro_precision=metrics["macro_precision"],
        macro_recall=metrics["macro_recall"],
        macro_f1=metrics["macro_f1"],
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        confusion_matrix_path=matrix_path,
    )


def save_evaluation_summary(
    results: Sequence[EvaluationResult], destination: str | Path
) -> Path:
    """Save a flat CSV suitable for cross-model and cross-condition comparisons."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "condition", "sample_count", "accuracy", "macro_precision", "macro_recall", "macro_f1"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            writer.writerow({name: row[name] for name in fieldnames})
    return path


def plot_metric_comparison(
    results: Sequence[EvaluationResult], destination: str | Path, metric: str = "macro_f1"
) -> Path:
    """Plot grouped condition scores without implying unmeasured conclusions."""
    if metric not in {"accuracy", "macro_precision", "macro_recall", "macro_f1"}:
        raise ValueError(f"Unsupported comparison metric: {metric}")
    models = list(dict.fromkeys(result.model for result in results))
    conditions = list(dict.fromkeys(result.condition for result in results))
    if not models or not conditions:
        raise ValueError("At least one evaluation result is required")
    lookup = {(result.model, result.condition): getattr(result, metric) for result in results}
    x = np.arange(len(conditions))
    width = 0.8 / len(models)
    figure, axis = plt.subplots(figsize=(max(8, len(conditions) * 1.2), 5))
    for model_index, model_name in enumerate(models):
        values = [lookup.get((model_name, condition), np.nan) for condition in conditions]
        offset = (model_index - (len(models) - 1) / 2) * width
        axis.bar(x + offset, values, width, label=model_name)
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_ylim(0, 1)
    axis.set_xticks(x, conditions, rotation=35, ha="right")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


"""Shared supervised training and validation loops for both classifiers."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .utils import save_json_atomic


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_accuracy: float
    elapsed_seconds: float


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_val_accuracy: float
    checkpoint_path: Path
    history_path: Path
    history: tuple[EpochMetrics, ...]


def load_checkpoint(
    model: nn.Module, checkpoint_path: str | Path, device: torch.device
) -> dict:
    """Load a trusted project checkpoint into an already constructed architecture."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with PyTorch versions predating weights_only.
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid project checkpoint: {path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return checkpoint


def _run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.train()
    # Frozen parameters do not receive gradients, but train() would still update
    # BatchNorm running statistics. Keep the pretrained feature extractor fixed.
    if getattr(model, "features_frozen", False):
        model.features.eval()
    total_loss = 0.0
    sample_count = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("Training loader contains no samples")
    return total_loss / sample_count


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    sample_count = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("Validation loader contains no samples")
    return total_loss / sample_count, correct / sample_count


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_accuracy: float,
    destination: Path,
    config: ExperimentConfig,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "validation_accuracy": validation_accuracy,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
        },
        temporary,
    )
    temporary.replace(destination)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: str | Path,
) -> TrainingResult:
    """Train with cross-entropy and retain the checkpoint with highest validation accuracy."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("Model has no trainable parameters")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    history: list[EpochMetrics] = []
    best_accuracy = float("-inf")
    best_epoch = 0
    checkpoint_path = destination / "best_model.pt"
    history_path = destination / "training_history.json"
    for epoch in range(1, config.training.epochs + 1):
        started = time.perf_counter()
        train_loss = _run_training_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=config.training.max_train_batches,
        )
        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion,
            device,
            max_batches=config.training.max_validation_batches,
        )
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
            elapsed_seconds=time.perf_counter() - started,
        )
        history.append(metrics)
        save_json_atomic([asdict(item) for item in history], history_path)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            _save_checkpoint(
                model, optimizer, epoch, validation_accuracy, checkpoint_path, config
            )
        print(
            f"Epoch {epoch:02d}/{config.training.epochs}: "
            f"train_loss={train_loss:.4f} val_loss={validation_loss:.4f} "
            f"val_accuracy={validation_accuracy:.4f}"
        )
    return TrainingResult(
        best_epoch=best_epoch,
        best_val_accuracy=best_accuracy,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        history=tuple(history),
    )


"""Typed experiment configuration loaded from a small YAML file."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class DataConfig:
    dataset_root: str = "/kaggle/input/intel-image-classification"
    train_dir: str | None = None
    test_dir: str | None = None
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 2
    validation_fraction: float = 0.15
    pin_memory: bool = True


@dataclass
class MobileNetConfig:
    pretrained: bool = True
    freeze_features: bool = True


@dataclass
class ModelConfig:
    num_classes: int = 6
    baseline_channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    dropout: float = 0.3
    mobilenet: MobileNetConfig = field(default_factory=MobileNetConfig)


@dataclass
class TrainingConfig:
    epochs: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_train_batches: int | None = None
    max_validation_batches: int | None = None


@dataclass
class DegradationConfig:
    gaussian_blur_sigmas: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    brightness_factors: list[float] = field(default_factory=lambda: [0.6, 1.4])
    jpeg_qualities: list[int] = field(default_factory=lambda: [90, 50, 20])


@dataclass
class EvaluationConfig:
    error_examples: int = 12
    error_conditions: list[str] = field(
        default_factory=lambda: [
            "clean",
            "gaussian_blur_sigma_4",
            "brightness_factor_0.6",
            "brightness_factor_1.4",
            "jpeg_quality_20",
        ]
    )


@dataclass
class BenchmarkConfig:
    batch_size: int = 1
    warmup_iterations: int = 10
    timed_iterations: int = 50


@dataclass
class ExperimentConfig:
    seed: int = 42
    output_dir: str = "/kaggle/working/image-robustness-results"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    degradations: DegradationConfig = field(default_factory=DegradationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.data.image_size <= 0 or self.data.batch_size <= 0:
            raise ValueError("image_size and batch_size must be positive")
        if self.data.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not 0.0 < self.data.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.model.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not self.model.baseline_channels or any(c <= 0 for c in self.model.baseline_channels):
            raise ValueError("baseline_channels must contain positive integers")
        if not 0.0 <= self.model.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.training.epochs <= 0 or self.training.learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        for name in ("max_train_batches", "max_validation_batches"):
            value = getattr(self.training, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or null")
        if any(s <= 0 for s in self.degradations.gaussian_blur_sigmas):
            raise ValueError("Gaussian blur sigmas must be positive")
        if any(f <= 0 for f in self.degradations.brightness_factors):
            raise ValueError("brightness factors must be positive")
        if any(not 1 <= q <= 100 for q in self.degradations.jpeg_qualities):
            raise ValueError("JPEG qualities must be between 1 and 100")
        if self.evaluation.error_examples <= 0:
            raise ValueError("error_examples must be positive")
        if not self.evaluation.error_conditions:
            raise ValueError("error_conditions cannot be empty")
        if self.benchmark.batch_size <= 0:
            raise ValueError("benchmark batch_size must be positive")
        if self.benchmark.warmup_iterations < 0 or self.benchmark.timed_iterations <= 0:
            raise ValueError("benchmark iterations are invalid")


def _merge_dataclass(instance: Any, values: Mapping[str, Any], prefix: str = "") -> Any:
    """Apply mapping values recursively while rejecting unknown keys."""
    fields = instance.__dataclass_fields__
    for key, value in values.items():
        if key not in fields:
            raise KeyError(f"Unknown configuration key: {prefix}{key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__"):
            if not isinstance(value, Mapping):
                raise TypeError(f"Configuration section {prefix}{key} must be a mapping")
            _merge_dataclass(current, value, prefix=f"{prefix}{key}.")
        else:
            setattr(instance, key, value)
    return instance


def _apply_overrides(raw: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target = raw
        for part in parts[:-1]:
            child = target.setdefault(part, {})
            if not isinstance(child, dict):
                raise TypeError(f"Cannot override nested key under {part!r}")
            target = child
        target[parts[-1]] = value


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> ExperimentConfig:
    """Load YAML plus optional dotted-key overrides into a validated config."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("The configuration file must contain a top-level mapping")
    if overrides:
        _apply_overrides(raw, overrides)
    config = _merge_dataclass(ExperimentConfig(), raw)
    config.validate()
    return config


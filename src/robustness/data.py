"""Dataset discovery, preprocessing, stable splitting, and DataLoader creation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .config import ExperimentConfig
from .degradations import ImageDegradation

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetPaths:
    train: Path
    test: Path


@dataclass(frozen=True)
class DataSummary:
    train_directory: str
    test_directory: str
    class_names: tuple[str, ...]
    train_samples: int
    validation_samples: int
    test_samples: int
    train_per_class: dict[str, int]
    validation_per_class: dict[str, int]
    test_per_class: dict[str, int]


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: tuple[str, ...]
    class_to_idx: dict[str, int]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_dataset: datasets.ImageFolder
    evaluation_transform: Callable
    summary: DataSummary


class TransformSubset(Dataset):
    """A view over ImageFolder samples with its own transform and stable indices."""

    def __init__(
        self,
        base: datasets.ImageFolder,
        indices: Sequence[int],
        transform: Callable,
        degradation: ImageDegradation | None = None,
    ) -> None:
        self.base = base
        self.indices = tuple(int(index) for index in indices)
        self.transform = transform
        self.degradation = degradation
        self.classes = base.classes
        self.class_to_idx = base.class_to_idx
        self.targets = [base.targets[index] for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int]:
        sample_index = self.indices[position]
        path, label = self.base.samples[sample_index]
        with Image.open(path) as image_file:
            image = image_file.convert("RGB")
        if self.degradation is not None:
            image = self.degradation(image)
        return self.transform(image), label


def _is_imagefolder_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    class_dirs = [child for child in path.iterdir() if child.is_dir() and not child.name.startswith(".")]
    return len(class_dirs) >= 2


def _resolve_explicit(root: Path, value: str | None, name: str) -> Path | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    if not _is_imagefolder_root(candidate):
        raise FileNotFoundError(
            f"Configured {name} directory is missing or does not contain class subdirectories: {candidate}"
        )
    return candidate.resolve()


def discover_dataset_paths(
    dataset_root: str | Path,
    train_dir: str | None = None,
    test_dir: str | None = None,
) -> DatasetPaths:
    """Resolve explicit folders or common Intel dataset layouts with clear errors."""
    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}. Attach the Kaggle dataset and set data.dataset_root "
            "to its mounted directory under /kaggle/input."
        )
    explicit_train = _resolve_explicit(root, train_dir, "training")
    explicit_test = _resolve_explicit(root, test_dir, "test")
    if (explicit_train is None) != (explicit_test is None):
        raise ValueError("Set both data.train_dir and data.test_dir, or leave both null for discovery")
    if explicit_train is not None and explicit_test is not None:
        return DatasetPaths(explicit_train, explicit_test)

    relative_pairs = (
        ("seg_train/seg_train", "seg_test/seg_test"),
        ("seg_train", "seg_test"),
        ("train", "test"),
    )
    for train_relative, test_relative in relative_pairs:
        candidate_train, candidate_test = root / train_relative, root / test_relative
        if _is_imagefolder_root(candidate_train) and _is_imagefolder_root(candidate_test):
            return DatasetPaths(candidate_train.resolve(), candidate_test.resolve())
    checked = ", ".join(f"({a}, {b})" for a, b in relative_pairs)
    raise FileNotFoundError(
        f"Could not discover training and test folders under {root}. Checked {checked}. "
        "Inspect the Kaggle mount, then set both data.train_dir and data.test_dir explicitly."
    )


def stratified_split_indices(
    targets: Sequence[int], validation_fraction: float, seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split every class reproducibly so validation cannot borrow test samples."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(targets):
        grouped.setdefault(int(label), []).append(index)
    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for label, indices in sorted(grouped.items()):
        if len(indices) < 2:
            raise ValueError(f"Class index {label} needs at least two training images for a split")
        order = torch.randperm(len(indices), generator=generator).tolist()
        validation_count = max(1, round(len(indices) * validation_fraction))
        validation_count = min(validation_count, len(indices) - 1)
        validation_indices.extend(indices[position] for position in order[:validation_count])
        train_indices.extend(indices[position] for position in order[validation_count:])
    return tuple(sorted(train_indices)), tuple(sorted(validation_indices))


def build_transforms(image_size: int) -> tuple[Callable, Callable]:
    """Return train augmentation and deterministic evaluation preprocessing."""
    normalization = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            # MobileNetV2's current torchvision weights use a 232 -> 224
            # resize/crop recipe. Scale that recipe when image_size changes.
            transforms.Resize(int(round(image_size * 232 / 224))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, evaluation_transform


def _named_counts(targets: Sequence[int], class_names: Sequence[str]) -> dict[str, int]:
    counts = Counter(int(target) for target in targets)
    return {name: counts.get(index, 0) for index, name in enumerate(class_names)}


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    import random
    import numpy as np

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloaders(config: ExperimentConfig) -> DataBundle:
    """Construct consistent train, validation, and test loaders without reading test labels for selection."""
    config.validate()
    paths = discover_dataset_paths(
        config.data.dataset_root, config.data.train_dir, config.data.test_dir
    )
    train_base = datasets.ImageFolder(paths.train)
    test_base = datasets.ImageFolder(paths.test)
    if train_base.class_to_idx != test_base.class_to_idx:
        raise ValueError(
            "Training/test class mappings differ. "
            f"train={train_base.class_to_idx}, test={test_base.class_to_idx}"
        )
    if len(train_base.classes) != config.model.num_classes:
        raise ValueError(
            f"Configuration expects {config.model.num_classes} classes, but dataset has "
            f"{len(train_base.classes)}: {train_base.classes}"
        )
    train_indices, validation_indices = stratified_split_indices(
        train_base.targets, config.data.validation_fraction, config.seed
    )
    train_transform, evaluation_transform = build_transforms(config.data.image_size)
    training_dataset = TransformSubset(train_base, train_indices, train_transform)
    validation_dataset = TransformSubset(train_base, validation_indices, evaluation_transform)
    test_indices = tuple(range(len(test_base)))
    test_dataset_view = TransformSubset(test_base, test_indices, evaluation_transform)

    generator = torch.Generator().manual_seed(config.seed)
    common = dict(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
    )
    train_loader = DataLoader(training_dataset, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(validation_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset_view, shuffle=False, **common)
    summary = DataSummary(
        train_directory=str(paths.train),
        test_directory=str(paths.test),
        class_names=tuple(train_base.classes),
        train_samples=len(training_dataset),
        validation_samples=len(validation_dataset),
        test_samples=len(test_base),
        train_per_class=_named_counts(training_dataset.targets, train_base.classes),
        validation_per_class=_named_counts(validation_dataset.targets, train_base.classes),
        test_per_class=_named_counts(test_base.targets, train_base.classes),
    )
    return DataBundle(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        class_names=tuple(train_base.classes),
        class_to_idx=dict(train_base.class_to_idx),
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_dataset=test_base,
        evaluation_transform=evaluation_transform,
        summary=summary,
    )


def build_degraded_test_loader(
    bundle: DataBundle,
    degradation: ImageDegradation | None,
    config: ExperimentConfig,
) -> DataLoader:
    """Create a clean/corrupted view over exactly the same ordered test samples."""
    dataset = TransformSubset(
        bundle.test_dataset,
        range(len(bundle.test_dataset)),
        bundle.evaluation_transform,
        degradation=degradation,
    )
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
    )


def format_data_summary(summary: DataSummary) -> str:
    lines = [
        f"Training directory: {summary.train_directory}",
        f"Test directory: {summary.test_directory}",
        f"Classes ({len(summary.class_names)}): {', '.join(summary.class_names)}",
        f"Samples: train={summary.train_samples}, validation={summary.validation_samples}, test={summary.test_samples}",
        f"Train per class: {summary.train_per_class}",
        f"Validation per class: {summary.validation_per_class}",
        f"Test per class: {summary.test_per_class}",
    ]
    return "\n".join(lines)


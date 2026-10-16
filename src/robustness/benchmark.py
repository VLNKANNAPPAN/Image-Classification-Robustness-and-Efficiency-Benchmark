"""Inference-only latency and model-size benchmarking."""

from __future__ import annotations

import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .utils import save_json_atomic


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    device: str
    batch_size: int
    input_shape: tuple[int, ...]
    warmup_iterations: int
    timed_iterations: int
    median_batch_latency_ms: float
    median_per_image_latency_ms: float
    iqr_batch_latency_ms: float
    throughput_images_per_second: float
    total_parameters: int
    trainable_parameters: int
    checkpoint_size_mb: float | None
    torch_version: str
    python_version: str
    processor: str
    cuda_device: str | None


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def checkpoint_size_mb(path: str | Path | None) -> float | None:
    if path is None:
        return None
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint.stat().st_size / (1024 * 1024)


@torch.inference_mode()
def benchmark_model(
    model: nn.Module,
    input_batch: torch.Tensor,
    device: torch.device,
    model_name: str,
    warmup_iterations: int = 10,
    timed_iterations: int = 50,
    checkpoint_path: str | Path | None = None,
) -> BenchmarkResult:
    """Measure model execution only; callers must preload and preprocess input_batch."""
    if input_batch.ndim != 4 or input_batch.shape[0] == 0:
        raise ValueError("input_batch must be a non-empty NCHW tensor")
    if warmup_iterations < 0 or timed_iterations <= 0:
        raise ValueError("warmup_iterations must be non-negative and timed_iterations positive")
    model = model.to(device)
    model.eval()
    inputs = input_batch.to(device)
    for _ in range(warmup_iterations):
        model(inputs)
    _synchronize(device)
    elapsed_ms: list[float] = []
    for _ in range(timed_iterations):
        _synchronize(device)
        started = time.perf_counter_ns()
        model(inputs)
        _synchronize(device)
        elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    median = statistics.median(elapsed_ms)
    quartiles = (
        statistics.quantiles(elapsed_ms, n=4, method="inclusive")
        if len(elapsed_ms) > 1
        else [median, median, median]
    )
    batch_size = input_batch.shape[0]
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return BenchmarkResult(
        model=model_name,
        device=str(device),
        batch_size=batch_size,
        input_shape=tuple(input_batch.shape),
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
        median_batch_latency_ms=median,
        median_per_image_latency_ms=median / batch_size,
        iqr_batch_latency_ms=quartiles[2] - quartiles[0],
        throughput_images_per_second=batch_size * 1000 / median,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        checkpoint_size_mb=checkpoint_size_mb(checkpoint_path),
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        processor=platform.processor() or platform.machine(),
        cuda_device=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    )


def save_benchmark(result: BenchmarkResult, destination: str | Path) -> Path:
    return save_json_atomic(asdict(result), destination)


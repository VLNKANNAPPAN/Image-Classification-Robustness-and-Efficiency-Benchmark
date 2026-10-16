"""Run phases 3–8 reproducibly in a Kaggle runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from robustness.benchmark import benchmark_model, save_benchmark
from robustness.config import load_config
from robustness.data import build_dataloaders, build_degraded_test_loader, format_data_summary
from robustness.degradations import build_degradations
from robustness.error_analysis import collect_misclassifications, save_error_gallery
from robustness.evaluate import (
    EvaluationResult,
    evaluate_model,
    plot_metric_comparison,
    save_evaluation_summary,
)
from robustness.models import build_model
from robustness.train import load_checkpoint, train_model
from robustness.utils import prepare_experiment, save_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--models", nargs="+", choices=("baseline", "mobilenet_v2"), default=("baseline", "mobilenet_v2"))
    parser.add_argument("--skip-training", action="store_true", help="Load existing best_model.pt files")
    parser.add_argument("--smoke-test", action="store_true", help="Use one epoch and a few batches; results are not final")
    return parser.parse_args()


def _config_overrides(args: argparse.Namespace) -> dict:
    overrides = {}
    if args.dataset_root:
        overrides["data.dataset_root"] = args.dataset_root
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.smoke_test:
        overrides.update(
            {
                "training.epochs": 1,
                "training.max_train_batches": 10,
                "training.max_validation_batches": 3,
                "benchmark.warmup_iterations": 2,
                "benchmark.timed_iterations": 5,
            }
        )
    return overrides


def main() -> None:
    args = parse_args()
    config = load_config(args.config, _config_overrides(args))
    device, output_root = prepare_experiment(config)
    data = build_dataloaders(config)
    print(format_data_summary(data.summary))
    print(f"Training/evaluation device: {device}")
    save_json_atomic(asdict(config), output_root / "resolved_config.json")
    conditions = build_degradations(config.degradations)
    unknown_error_conditions = set(config.evaluation.error_conditions) - set(conditions)
    if unknown_error_conditions:
        raise ValueError(f"Unknown evaluation.error_conditions: {sorted(unknown_error_conditions)}")
    evaluation_results: list[EvaluationResult] = []

    for model_name in args.models:
        model_dir = output_root / model_name
        checkpoint_path = model_dir / "best_model.pt"
        model = build_model(
            model_name,
            config,
            len(data.class_names),
            pretrained_override=False if args.skip_training else None,
        )
        if args.skip_training:
            load_checkpoint(model, checkpoint_path, device)
        else:
            training_result = train_model(
                model, data.train, data.val, config, device, model_dir
            )
            checkpoint_path = training_result.checkpoint_path
            load_checkpoint(model, checkpoint_path, device)

        for condition_name, degradation in conditions.items():
            loader = build_degraded_test_loader(data, degradation, config)
            result = evaluate_model(
                model,
                loader,
                data.class_names,
                device,
                model_name,
                condition_name,
                output_root / "metrics",
            )
            evaluation_results.append(result)
            print(
                f"{model_name}/{condition_name}: accuracy={result.accuracy:.4f}, "
                f"macro_f1={result.macro_f1:.4f}"
            )
            if condition_name in config.evaluation.error_conditions:
                errors = collect_misclassifications(
                    model, loader, device, limit=config.evaluation.error_examples
                )
                save_error_gallery(
                    errors,
                    data.class_names,
                    model_name,
                    condition_name,
                    output_root / "figures" / "errors" / model_name / condition_name,
                )

        clean_images, _ = next(iter(data.test))
        benchmark_batch = clean_images[: config.benchmark.batch_size]
        if benchmark_batch.shape[0] != config.benchmark.batch_size:
            raise ValueError("Test loader did not provide the configured benchmark batch size")
        benchmark_devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            benchmark_devices.append(torch.device("cuda"))
        for benchmark_device in benchmark_devices:
            benchmark = benchmark_model(
                model,
                benchmark_batch,
                benchmark_device,
                model_name,
                config.benchmark.warmup_iterations,
                config.benchmark.timed_iterations,
                checkpoint_path,
            )
            save_benchmark(
                benchmark,
                output_root / "metrics" / "benchmarks" / f"{model_name}_{benchmark_device.type}.json",
            )
            print(
                f"{model_name}/{benchmark_device}: median per-image latency "
                f"{benchmark.median_per_image_latency_ms:.3f} ms"
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_evaluation_summary(evaluation_results, output_root / "metrics" / "evaluation_summary.csv")
    plot_metric_comparison(
        evaluation_results, output_root / "figures" / "macro_f1_comparison.png", "macro_f1"
    )
    plot_metric_comparison(
        evaluation_results, output_root / "figures" / "accuracy_comparison.png", "accuracy"
    )
    if args.smoke_test:
        print("SMOKE TEST ONLY: do not report these values as final experimental results.")


if __name__ == "__main__":
    main()


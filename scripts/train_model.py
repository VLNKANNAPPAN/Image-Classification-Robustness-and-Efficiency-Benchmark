"""Kaggle-oriented command-line entry point for training one configured model."""

from __future__ import annotations

import argparse
from pathlib import Path

from robustness.config import load_config
from robustness.data import build_dataloaders, format_data_summary
from robustness.models import build_model, count_parameters
from robustness.train import train_model
from robustness.utils import prepare_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=("baseline", "mobilenet_v2"), required=True)
    parser.add_argument("--dataset-root", help="Override data.dataset_root")
    parser.add_argument("--output-dir", help="Override output_dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.dataset_root:
        overrides["data.dataset_root"] = args.dataset_root
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    config = load_config(args.config, overrides=overrides)
    device, output_root = prepare_experiment(config)
    data = build_dataloaders(config)
    print(format_data_summary(data.summary))
    print(f"Selected device: {device}")
    model = build_model(args.model, config, num_classes=len(data.class_names))
    print(
        f"Parameters: total={count_parameters(model):,}, "
        f"trainable={count_parameters(model, trainable_only=True):,}"
    )
    result = train_model(
        model,
        data.train,
        data.val,
        config,
        device,
        Path(output_root) / args.model,
    )
    print(
        f"Best validation accuracy={result.best_val_accuracy:.4f} "
        f"at epoch={result.best_epoch}; checkpoint={result.checkpoint_path}"
    )


if __name__ == "__main__":
    main()


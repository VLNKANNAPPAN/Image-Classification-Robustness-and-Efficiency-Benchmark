# Image Classification Robustness and Efficiency Benchmark

This project compares a small convolutional neural network trained from scratch with a pretrained MobileNetV2 on the six-class Intel Image Classification dataset. The study is designed for Kaggle GPU notebooks and asks two questions: how much do controlled image corruptions change classification quality, and what accuracy/efficiency trade-offs do the two models make?

> The repository provides the complete training, robustness evaluation, efficiency benchmarking, failure-analysis, and reporting workflow.

## Experiment design

The supplied Intel training split is divided reproducibly into training and validation subsets. Training data updates model parameters; validation data selects the best checkpoint; the supplied test split is reserved for final evaluation. Every model uses the same split indices, image size, class ordering, and test examples. The initial preprocessing contract uses ImageNet normalization for both models, which is required by pretrained MobileNetV2 and gives the baseline an identical numeric input representation.

Data flows through the system as follows:

```text
Kaggle dataset -> path discovery -> ImageFolder/class validation
               -> reproducible train/validation indices
               -> train-only augmentation -> model -> loss/backprop/checkpoint
               -> clean or degraded test view -> metrics and confusion matrices
               -> latency benchmark / failure analysis -> exported comparison tables
```

The implemented metrics are accuracy, macro precision/recall/F1, per-class metrics, confusion matrices, parameter/checkpoint size, and carefully measured CPU/GPU latency. Accuracy measures total correctness; macro metrics weight each class equally; confusion matrices expose which scene pairs are confused. The repository does not invent placeholder results.

Important confounders are controlled by keeping preprocessing, splits, test order, degradation parameters, and evaluation code fixed. Test-set leakage, choosing settings from test results, including decoding in only one model's latency, or changing normalization between clean and corrupted inputs would invalidate comparisons.

## Implemented modules

- `config.py`: typed YAML configuration and validation.
- `data.py`: dataset discovery, deterministic stratified validation splitting, transforms, loaders, and degraded test views.
- `models.py`: a readable baseline CNN and configurable MobileNetV2 transfer learning.
- `train.py`: shared training/validation loops, best-checkpoint selection, and history saving.
- `degradations.py`: deterministic Gaussian blur, brightness, and JPEG corruptions.
- `evaluate.py`: overall/per-class metrics, predictions, confusion matrices, and comparisons.
- `benchmark.py`: warmed-up repeated CPU/GPU inference timing and environment metadata.
- `error_analysis.py`: high-confidence error collection, manifests, and image galleries.
- `utils.py`: seeding, device selection, directory setup, and atomic JSON output.

The notebook is an orchestration layer; reusable logic remains in the package.

## Kaggle setup

1. Create a Kaggle notebook, enable a GPU accelerator, attach the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification), and upload or clone this repository.
2. From the repository directory, run `python -m pip install -e . --no-deps`. This registers the local `src` package without downloading or upgrading the Kaggle environment's libraries.
3. Inspect `/kaggle/input` and set `data.dataset_root` in `configs/default.yaml` (or override it in Python) to the directory containing the dataset folders.
4. Keep `output_dir` under `/kaggle/working`, because `/kaggle/input` is read-only.
5. Verify the runtime before training:

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

6. Load and inspect configuration/data without training:

```python
from robustness.config import load_config
from robustness.data import build_dataloaders, format_data_summary

cfg = load_config("configs/default.yaml", overrides={
    "data.dataset_root": "/kaggle/input/intel-image-classification",
    "output_dir": "/kaggle/working/image-robustness-results",
})
bundle = build_dataloaders(cfg)
print(format_data_summary(bundle.summary))
images, labels = next(iter(bundle.train))
print(images.shape, labels.shape, images.dtype)
```

The expected image batch shape is `[batch_size, 3, image_size, image_size]`. Dataset discovery supports common nested `seg_train/seg_train` and `seg_test/seg_test` layouts, but its printed discovery result must be checked against the actual Kaggle mount.

## Training example

```python
from robustness.models import build_model
from robustness.train import train_model
from robustness.utils import prepare_experiment

device, output_dir = prepare_experiment(cfg)
model = build_model("baseline", cfg, num_classes=len(bundle.class_names))
result = train_model(model, bundle.train, bundle.val, cfg, device, output_dir / "baseline")
print(result.best_epoch, result.best_val_accuracy, result.checkpoint_path)
```

For MobileNetV2 use `build_model("mobilenet_v2", ...)`. Its pretrained weights may need a one-time internet download unless already cached in the Kaggle image. If internet is unavailable, attach the weights as a Kaggle dataset/cache or explicitly set `model.mobilenet.pretrained: false`; the latter changes the experiment from transfer learning and must be documented.

For the complete pipeline, first run a smoke test:

```bash
python scripts/run_experiments.py \
  --dataset-root /kaggle/input/intel-image-classification \
  --output-dir /kaggle/working/image-robustness-results \
  --smoke-test
```

After that succeeds, remove `--smoke-test` for the actual experiment. Existing best checkpoints can be evaluated without retraining by adding `--skip-training`. Never present smoke-test numbers as final results.

## Degraded test views

Corruptions are applied to an RGB copy before the common resize, tensor conversion, and normalization steps. This preserves labels, ordering, dimensions, and preprocessing across conditions:

```python
from robustness.data import build_degraded_test_loader
from robustness.degradations import build_degradations

for condition, degradation in build_degradations(cfg.degradations).items():
    test_loader = build_degraded_test_loader(bundle, degradation, cfg)
    # Phase 6 will pass test_loader to the shared evaluation function.
```

## Current limitations

- Gaussian blur, global brightness scaling, and JPEG compression are controlled proxies, not a complete model of real cameras or identity-verification systems.
- Reproducible seeds reduce variation but do not guarantee bitwise-identical GPU training across all PyTorch/CUDA versions.
- The study uses one dataset split and one primary seed; repeated-seed confidence intervals would strengthen conclusions.

## Metrics and interpretation

For class counts `TP`, `FP`, and `FN`:

- Accuracy is the fraction of all predictions that are correct.
- Precision is `TP / (TP + FP)` and asks how often a predicted class is correct.
- Recall is `TP / (TP + FN)` and asks how many members of a true class were found.
- F1 is the harmonic mean `2PR / (P + R)`.
- Macro metrics calculate each class independently and average the six values, so a weak class cannot be hidden by a strong or more frequent class.

Zero-denominator cases are explicitly scored as zero. Each model/condition saves the sample count, predictions, per-class report, and confusion matrix. A confusion matrix uses true labels as rows and predicted labels as columns.

Latency timing receives an already decoded, resized, normalized tensor. Warm-up calls are excluded; CUDA is synchronized around every measurement; the median and interquartile range are reported in milliseconds. Batch size, tensor shape, device, runtime, parameter count, checkpoint size, and throughput are stored with the result. CPU and GPU values are never treated as interchangeable.

## Generated artifacts

```text
/kaggle/working/image-robustness-results/
├── resolved_config.json
├── baseline/ and mobilenet_v2/
│   ├── best_model.pt
│   └── training_history.json
├── metrics/
│   ├── evaluation_summary.csv
│   ├── benchmarks/*.json
│   └── <model>/<condition>/{metrics.json,predictions.csv,confusion_matrix.png}
└── figures/
    ├── accuracy_comparison.png
    ├── macro_f1_comparison.png
    └── errors/<model>/<condition>/{misclassifications.csv,misclassifications.png}
```

## Reporting results

Use `evaluation_summary.csv` and the benchmark JSON files as the source of truth for reporting. Include the runtime hardware, resolved configuration, clean scores, absolute degradation drops, macro/per-class behavior, latency variability, and limitations alongside every conclusion.

## Failure-analysis discipline

The gallery prioritizes wrong predictions with high model confidence because these can reveal systematic failures. A pattern visible in several images is an observation; statements about its cause remain hypotheses. For example, “mountains were often predicted as glaciers under blur” can be measured, while “the network relies on snow edges” requires additional evidence. Confidence is a softmax score, not a calibrated probability or proof of the model’s reasoning.

## Reproducibility checklist

- Save the resolved configuration and library/GPU versions with the artifacts.
- Keep split seed, class mapping, input size, and condition parameters fixed.
- Select checkpoints and hyperparameters using validation data only.
- Run every condition over the same ordered test samples.
- Compare latency only when device, batch, shape, warm-up, and timing boundary match.
- Preserve full-run outputs separately from smoke-test outputs.


# Interview and result-interpretation guide

Use this guide only after completing a full Kaggle run. Replace bracketed fields with values copied from saved artifacts; do not estimate them.

## Two-minute project explanation

I built a reproducible PyTorch experiment comparing a compact CNN trained from scratch with a pretrained MobileNetV2 on six Intel scene classes. I separated the supplied training images into deterministic stratified training and validation subsets and reserved the supplied test set for final evaluation. Both models used the same 224-pixel input contract and ImageNet normalization. I selected checkpoints by validation accuracy, then evaluated the same ordered test examples under clean, Gaussian blur, brightness, and JPEG conditions. I recorded overall and per-class classification metrics, confusion matrices, high-confidence errors, model size, and warmed-up inference latency measured separately from decoding and preprocessing. The main measured trade-off was [insert verified comparison]. The largest observed limitation was [insert verified limitation], and a useful follow-up would be [experiment motivated by the evidence].

## Result worksheet

- Full-run configuration and seed: [resolved_config.json]
- Kaggle GPU/runtime: [benchmark JSON]
- Baseline clean accuracy / macro F1: [values]
- MobileNetV2 clean accuracy / macro F1: [values]
- Largest absolute accuracy and macro-F1 drops: [model, condition, values]
- Most affected classes and confusion pairs: [from metrics/confusion matrices]
- CPU median per-image latency and IQR: [values]
- Parameter and checkpoint sizes: [values]
- High-confidence error patterns directly observed: [observations]
- Alternative explanations or uncertainty: [hypotheses]

## Questions to be ready for

1. Why was the validation split created from training data instead of using the test set?
2. Why does MobileNetV2 require ImageNet normalization when pretrained weights are used?
3. Why does `CrossEntropyLoss` receive logits rather than softmax probabilities?
4. What do convolution, pooling, and receptive field mean in the baseline network?
5. What changes when MobileNet features are frozen, and what still changes in batch normalization?
6. Why can macro F1 disagree with accuracy?
7. How do blur, brightness, and JPEG artifacts shift the input distribution?
8. Why are these corruptions incomplete proxies for production camera conditions?
9. Why are decoding and data loading excluded from the model latency measurement?
10. Why are warm-up and CUDA synchronization necessary?
11. Why might parameter count and latency fail to scale proportionally?
12. What evidence would be needed before making a causal claim about a failure mode?

## Resume bullet template

- Built a reproducible PyTorch robustness benchmark comparing a compact CNN and transfer-learned MobileNetV2 across clean and controlled image-degradation conditions, measuring [verified metrics] on [verified test count] images.
- Implemented leakage-safe stratified validation, reusable evaluation, confusion-matrix/error analysis, and synchronized CPU/GPU latency benchmarking; observed [verified quantitative result].

Do not use these bullets until every bracketed statement is backed by saved full-run artifacts.

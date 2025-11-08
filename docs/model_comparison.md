# Model Comparison Guide

This document outlines how to benchmark the existing YOLO11 ASL detector against two alternative pipelines:

1. **MediaPipe Hands + Lightweight Classifier (PyTorch)** – a keypoint-based baseline optimised for CPU inference.
2. **TensorFlow Object Detection API** – a two-stage experiment using EfficientDet/SSD-style detectors.

The goal is to quantify accuracy, latency, and operational overhead to justify YOLO11 as the production model.

## Common Setup

- Use the dataset defined in `sign_language/training/asl_letters_v2/data.yaml`. Keep the same train/val/test splits for all models.
- Collect the following metrics for each approach:
  - mAP@0.5 (for detectors) or macro F1/accuracy for classifiers.
  - Per-class accuracy / confusion matrix.
  - Inference latency (ms/frame) on the target device (e.g., MacBook Air M3 GPU/CPU).
  - Model size on disk and peak memory usage.
- Record results in a shared experiment tracker (Weights & Biases, MLflow, spreadsheet, etc.).
- Use identical augmentations where possible (colour jitter, affine transforms) to maintain fairness.

### Requirements

Install optional dependencies:

```bash
pip install -r requirements-baselines.txt
```

## Baseline 1 – MediaPipe Hands + MLP Classifier

This baseline extracts 21 hand landmarks via MediaPipe and trains a small MLP to classify letters.

### Training

```bash
python -m sign_language.training.baselines.mediapipe_classifier \
  --data sign_language/training/asl_letters_v2/data.yaml \
  --epochs 50 \
  --batch-size 256 \
  --learning-rate 1e-3 \
  --output sign_language/training/runs/mediapipe_baseline
```

Artifacts:

- `sign_language/training/runs/mediapipe_baseline/best_classifier.pt`
- `sign_language/training/runs/mediapipe_baseline/metrics.json`

Metrics to capture: overall accuracy, macro F1, confusion matrix, MediaPipe failure rate (images without detected hands), and CPU inference latency (MediaPipe + MLP).

## Baseline 2 – TorchVision Detectors (PyTorch)

This experiment fine-tunes detectors such as Faster R-CNN or RetinaNet using TorchVision’s detection zoo.

### Training

```bash
python -m sign_language.training.baselines.torchvision_detector \
  --data sign_language/training/asl_letters_v2/data.yaml \
  --model fasterrcnn_mobilenet_v3_large_fpn \
  --epochs 50 \
  --batch-size 8 \
  --device cpu \
  --output sign_language/training/runs/torchvision_detector
```

To test RetinaNet instead:

```bash
python -m sign_language.training.baselines.torchvision_detector \
  --model retinanet_resnet50_fpn
```

Artifacts:

- `sign_language/training/runs/torchvision_detector/best_model.pt`
- `sign_language/training/runs/torchvision_detector/metrics.json`

Record evaluation metrics (mAP@0.5, macro F1, per-class precision/recall) and measure inference latency using TorchVision’s inference API on CPU, CUDA, or MPS as appropriate.

## Presenting Results

Create a comparison table summarising:

| Model | Accuracy / mAP | Macro F1 | Latency (ms) | Model Size | Notes |
|-------|----------------|----------|--------------|------------|-------|
| YOLO11n (current) | … | … | … | … | Strong balance of accuracy & speed |
| MediaPipe + MLP | … | … | … | … | Dependent on landmark detection, CPU-friendly |
| TorchVision (Faster R-CNN/RetinaNet) | … | … | … | … | Same PyTorch stack, slower but informative baseline |

Use the table plus qualitative observations (operational complexity, dependency count, integration effort) to motivate continuing with YOLO11 for production.

### YOLO metrics + automated comparison

Generate YOLO metrics in the same JSON format as the baselines:

```bash
python tools/generate_yolo_metrics.py \
  --weights models/asl-sign-detector.pt \
  --data sign_language/training/asl_letters_v2/data.yaml \
  --split test \
  --device mps
```

### Automated comparison script

After training each baseline you can generate a unified report:

```bash
python tools/compare_metrics.py \
  --metrics \
    "YOLO11=sign_language/training/runs/yolo_eval/metrics.json" \
    "MediaPipe=sign_language/training/runs/mediapipe_baseline/metrics.json" \
    "TorchVision=sign_language/training/runs/torchvision_detector/metrics.json" \
  --output-dir sign_language/training/runs/comparison
```

This script prints a table to the console and saves:

- `comparison_table.md` – Markdown summary of mAP@0.5, macro F1, precision/recall averages.
- `comparison_plot.png` – Bar chart comparing key metrics across models.

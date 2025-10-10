# Model Comparison Guide

This document outlines how to benchmark the existing YOLO11 ASL detector against two alternative pipelines:

1. **MediaPipe Hands + Lightweight Classifier (PyTorch)** – a keypoint-based baseline optimised for CPU inference.
2. **TensorFlow Object Detection API** – a two-stage experiment using EfficientDet/SSD-style detectors.

The goal is to quantify accuracy, latency, and operational overhead to justify YOLO11 as the production model.

## Common Setup

- Use the dataset defined in `sign_language/training/asl_letters/data.yaml`. Keep the same train/val/test splits for all models.
- Collect the following metrics for each approach:
  - mAP@0.5 (for detectors) or macro F1/accuracy for classifiers.
  - Per-class accuracy / confusion matrix.
  - Inference latency (ms/frame) on the target device (e.g., MacBook Air M3 GPU/CPU).
  - Model size on disk and peak memory usage.
- Record results in a shared experiment tracker (Weights & Biases, MLflow, spreadsheet, etc.).
- Use identical augmentations where possible (colour jitter, affine transforms) to maintain fairness.

## Baseline 1 – MediaPipe Hands + MLP Classifier

This baseline extracts 21 hand landmarks via MediaPipe and trains a small MLP to classify letters.

### Requirements

Install optional dependencies:

```bash
pip install -r requirements-baselines.txt
```

### Training

```bash
python -m sign_language.training.baselines.mediapipe_classifier \
  --data sign_language/training/asl_letters/data.yaml \
  --epochs 60 \
  --batch-size 256 \
  --learning-rate 1e-3 \
  --output sign_language/training/runs/mediapipe_baseline
```

Artifacts:

- `sign_language/training/runs/mediapipe_baseline/best_classifier.pt`
- `sign_language/training/runs/mediapipe_baseline/metrics.json`

Metrics to capture: overall accuracy, macro F1, confusion matrix, MediaPipe failure rate (images without detected hands), and CPU inference latency (MediaPipe + MLP).

## Baseline 2 – TensorFlow Object Detection API

This experiment converts the dataset into TFRecords and trains a detector such as EfficientDet-D0 or SSD MobileNet V2 using the TFOD API.

### Requirements

- TensorFlow 2.x (`pip install tensorflow`)
- TensorFlow Models repository with Object Detection API installed: <https://github.com/tensorflow/models/tree/master/research/object_detection>

Optional dependencies are listed in `requirements-baselines.txt`.

### Export TFRecords

```bash
python -m sign_language.training.baselines.tfod_export \
  --data sign_language/training/asl_letters/data.yaml \
  --output sign_language/training/runs/tfod
```

Outputs:

- `train.record`, `val.record`, `test.record`
- `label_map.pbtxt`

### Configure Training

1. Copy a baseline pipeline config, e.g. `ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config`.
2. Update the config:
   - `fine_tune_checkpoint` to the chosen pretrained checkpoint.
   - `train_config.batch_size`, `num_steps`, and optimizer schedule.
   - Input readers to point at the new TFRecords and label map.

### Launch Training

```bash
python model_main_tf2.py \
  --model_dir=training/tfod_asl \
  --pipeline_config_path=path/to/updated_config.config
```

Run evaluation (optional during training):

```bash
python model_main_tf2.py \
  --model_dir=training/tfod_asl \
  --pipeline_config_path=path/to/updated_config.config \
  --checkpoint_dir=training/tfod_asl
```

Export the trained model:

```bash
python exporter_main_v2.py \
  --input_type image_tensor \
  --pipeline_config_path path/to/updated_config.config \
  --trained_checkpoint_dir training/tfod_asl \
  --output_directory training/tfod_asl/exported_model
```

Record evaluation metrics (mAP@0.5, precision/recall) and measure inference latency with the exported SavedModel using `tensorflow` or `tfjs` runners.

## Presenting Results

Create a comparison table summarising:

| Model | Accuracy / mAP | Macro F1 | Latency (ms) | Model Size | Notes |
|-------|----------------|----------|--------------|------------|-------|
| YOLO11n (current) | … | … | … | … | Strong balance of accuracy & speed |
| MediaPipe + MLP | … | … | … | … | Dependent on landmark detection, CPU-friendly |
| TFOD (EfficientDet/SSD) | … | … | … | … | Requires TFRecords and heavier toolchain |

Use the table plus qualitative observations (operational complexity, dependency count, integration effort) to motivate continuing with YOLO11 for production.

"""Run YOLO evaluation and store metrics in the baseline format."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

from ultralytics import YOLO

_LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate YOLO metrics JSON for comparison")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/asl-sign-detector.pt"),
        help="Path to trained YOLO weights",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("sign_language/training/asl_letters/data.yaml"),
        help="YOLO data.yaml describing the dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (train/val/test)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sign_language/training/runs/yolo_eval/metrics.json"),
        help="Destination metrics JSON path",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("sign_language/training/runs/yolo_eval"),
        help="Ultralytics project directory for eval artifacts",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="asl_test",
        help="Run name used within the project directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cpu, cuda, mps). Uses Ultralytics default if omitted.",
    )
    return parser.parse_args()


def _build_metrics(results) -> Dict[str, object]:
    box_metrics = results.box
    class_names = results.names

    metrics = {
        "map": float(box_metrics.map),
        "map_50": float(box_metrics.map50),
        "map_75": float(box_metrics.map75),
        "precision_per_class": box_metrics.mp.tolist(),
        "recall_per_class": box_metrics.mr.tolist(),
        "class_names": class_names,
    }

    precision_avg = float(box_metrics.mp.mean().item()) if hasattr(box_metrics.mp, "mean") else float(box_metrics.mp.mean())
    recall_avg = float(box_metrics.mr.mean().item()) if hasattr(box_metrics.mr, "mean") else float(box_metrics.mr.mean())
    if precision_avg + recall_avg > 0:
        macro_f1 = 2 * precision_avg * recall_avg / (precision_avg + recall_avg)
    else:
        macro_f1 = 0.0
    metrics["macro_f1"] = macro_f1

    return metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found at {args.weights}")

    model = YOLO(str(args.weights))
    _LOGGER.info("Running YOLO validation on split '%s'", args.split)

    val_kwargs = {
        "data": str(args.data),
        "split": args.split,
        "save_json": True,
        "project": str(args.project),
        "name": args.name,
    }
    if args.device:
        val_kwargs["device"] = args.device

    results = model.val(**val_kwargs)
    metrics = _build_metrics(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _LOGGER.info("Metrics saved to %s", args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

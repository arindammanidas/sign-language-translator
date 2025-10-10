"""Training utilities for the ASL letters dataset."""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)


def _default_data_yaml() -> Path:
    return Path(__file__).resolve().parent / "data.yaml"


def _default_base_weights() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "yolo11n.pt"


def _default_output_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "models" / "asl-sign-detector.pt"


def _default_project_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "runs"


@dataclass(slots=True)
class ASLLettersTrainingConfig:
    """Configuration required to train the ASL letters YOLO model."""

    data_yaml: Path = _default_data_yaml()
    base_weights: Path = _default_base_weights()
    output_path: Path = _default_output_path()
    project_dir: Path = _default_project_dir()
    run_name: str = "asl_letters"
    epochs: int = 50
    batch_size: int = 16
    image_size: int = 640
    patience: int = 20
    seed: int = 42
    device: Optional[str] = None
    workers: Optional[int] = None


def train_asl_letters(config: ASLLettersTrainingConfig) -> Path:
    """Train a YOLO model on the ASL letters dataset and return the weights path."""

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:  # pragma: no cover - only raised when dependency missing
        raise RuntimeError(
            "ultralytics package is required. Install dependencies via requirements.txt before training."
        ) from exc

    if not config.data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found at {config.data_yaml}")

    if not config.base_weights.exists():
        raise FileNotFoundError(f"Base weights not found at {config.base_weights}")

    config.project_dir.mkdir(parents=True, exist_ok=True)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading base model from %s", config.base_weights)
    model = YOLO(str(config.base_weights))

    train_kwargs = {
        "data": str(config.data_yaml),
        "epochs": config.epochs,
        "imgsz": config.image_size,
        "batch": config.batch_size,
        "project": str(config.project_dir),
        "name": config.run_name,
        "patience": config.patience,
        "seed": config.seed,
    }

    if config.device:
        train_kwargs["device"] = config.device
    if config.workers is not None:
        train_kwargs["workers"] = config.workers

    LOGGER.info(
        "Starting training: epochs=%s, batch_size=%s, image_size=%s, patience=%s, device=%s",
        config.epochs,
        config.batch_size,
        config.image_size,
        config.patience,
        config.device or "auto",
    )

    results = model.train(**train_kwargs)
    best_weights = Path(results.save_dir) / "weights" / "best.pt"

    if not best_weights.exists():
        raise RuntimeError(
            f"Training finished but weights were not found at {best_weights}. Check Ultralytics run directory."
        )

    LOGGER.info("Copying best weights to %s", config.output_path)
    shutil.copy2(best_weights, config.output_path)

    return config.output_path


def _parse_args() -> argparse.Namespace:
    defaults = ASLLettersTrainingConfig()

    parser = argparse.ArgumentParser(description="Train YOLO on the ASL letters dataset.")
    parser.add_argument("--epochs", type=int, default=defaults.epochs, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="Training batch size")
    parser.add_argument("--image-size", type=int, default=defaults.image_size, help="Input image size")
    parser.add_argument("--patience", type=int, default=defaults.patience, help="Early stopping patience")
    parser.add_argument("--device", type=str, default=defaults.device, help="Torch device id (cpu, 0, 0,1, etc.)")
    parser.add_argument("--workers", type=int, default=defaults.workers, help="Number of dataloader workers")
    parser.add_argument(
        "--output",
        type=Path,
        default=defaults.output_path,
        help="Destination path for the trained weights",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=defaults.project_dir,
        help="Directory where Ultralytics should store training runs",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=defaults.run_name,
        help="Run name used within the training project directory",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=defaults.data_yaml,
        help="Path to the Ultralytics data.yaml configuration",
    )
    parser.add_argument(
        "--base-weights",
        type=Path,
        default=defaults.base_weights,
        help="Pretrained YOLO checkpoint to fine-tune",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    config = ASLLettersTrainingConfig(
        data_yaml=args.data,
        base_weights=args.base_weights,
        output_path=args.output,
        project_dir=args.project,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
    )

    weights_path = train_asl_letters(config)
    LOGGER.info("Training complete. Weights available at %s", weights_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

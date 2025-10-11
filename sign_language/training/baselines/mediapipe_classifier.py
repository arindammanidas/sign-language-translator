"""Train a MediaPipe-hands-based classifier as a baseline."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DatasetSplit:
    features: torch.Tensor
    labels: torch.Tensor


class KeypointDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, split: DatasetSplit) -> None:
        self._features = split.features
        self._labels = split.labels

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._labels.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - trivial
        return self._features[idx], self._labels[idx]


class KeypointClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple pass-through
        return self._model(x)


def _load_yaml(path: Path) -> Dict[str, object]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_split_dirs(data_cfg: Dict[str, object], key: str, base_dir: Path) -> Tuple[Path, Path]:
    value = data_cfg.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected '{key}' entry in data.yaml")

    rel_path = Path(value)
    candidates = []

    if rel_path.is_absolute():
        candidates.append(rel_path)
    else:
        candidates.append((base_dir / rel_path).resolve())
        if rel_path.parts and rel_path.parts[0] == "..":
            trimmed = Path(*rel_path.parts[1:])
            candidates.append((base_dir / trimmed).resolve())

    for candidate in candidates:
        images_dir = candidate
        if images_dir.exists():
            labels_dir = images_dir.parent / "labels"
            if labels_dir.exists():
                return images_dir, labels_dir

    raise FileNotFoundError(f"Unable to resolve '{key}' directories for {value}")


def _iter_image_label_pairs(images_dir: Path, labels_dir: Path) -> Iterable[Tuple[Path, Path]]:
    candidates: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        candidates.extend(images_dir.glob(pattern))
    for image_path in sorted(candidates):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        yield image_path, label_path


def _extract_keypoints(image_path: Path, hands: mp.solutions.hands.Hands) -> np.ndarray | None:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None

    # Take the most confident hand
    hand_landmarks = results.multi_hand_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

    # Normalise: translate wrist to origin, scale by max range to be size-invariant
    wrist = coords[0]
    coords -= wrist
    scale = np.linalg.norm(coords, axis=1).max()
    if scale > 0:
        coords /= scale

    return coords.flatten()


def _load_split_features(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    class_names: Sequence[str],
    max_samples: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
    features: List[np.ndarray] = []
    labels: List[int] = []

    iterator = list(_iter_image_label_pairs(images_dir, labels_dir))
    if max_samples is not None:
        iterator = iterator[:max_samples]

    for image_path, label_path in tqdm(iterator, desc=f"{split_name} extraction"):
        keypoints = _extract_keypoints(image_path, mp_hands)
        if keypoints is None:
            _LOGGER.debug("No hand detected in %s; skipping", image_path)
            continue

        with label_path.open("r", encoding="utf-8") as handle:
            entries = [line.strip().split() for line in handle if line.strip()]
        if not entries:
            continue
        class_id = int(entries[0][0])
        if class_id >= len(class_names):
            continue

        features.append(keypoints)
        labels.append(class_id)

    if not features:
        raise RuntimeError(f"No usable samples found for {split_name} split")

    return np.stack(features), np.array(labels, dtype=np.int64)


def _to_tensor_split(features: np.ndarray, labels: np.ndarray) -> DatasetSplit:
    return DatasetSplit(
        features=torch.from_numpy(features),
        labels=torch.from_numpy(labels),
    )


def _build_dataloaders(
    train: DatasetSplit,
    val: DatasetSplit,
    batch_size: int,
) -> Tuple[DataLoader[Tuple[torch.Tensor, torch.Tensor]], DataLoader[Tuple[torch.Tensor, torch.Tensor]]]:
    train_loader = DataLoader(KeypointDataset(train), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(KeypointDataset(val), batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def _evaluate(model: nn.Module, loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    correct = 0
    total = 0
    logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []

    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features.float())
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            logits_all.append(outputs.detach().cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    accuracy = correct / max(total, 1)
    logits_concat = np.concatenate(logits_all, axis=0)
    labels_concat = np.concatenate(labels_all, axis=0)
    return accuracy, logits_concat, labels_concat


def _train(
    model: nn.Module,
    train_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    epochs: int,
    learning_rate: float,
    best_weights_path: Path,
) -> None:
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            optimiser.zero_grad()
            logits = model(features.float())
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_acc, _, _ = _evaluate(model, val_loader)
        scheduler.step(1.0 - val_acc)
        _LOGGER.info("Epoch %s/%s - loss %.4f - val_acc %.3f", epoch, epochs, epoch_loss, val_acc)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_weights_path)


def run_pipeline(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)

    data_cfg_path = Path(args.data).resolve()
    data_cfg = _load_yaml(data_cfg_path)
    base_dir = data_cfg_path.parent
    class_names = data_cfg.get("names") or []
    if not isinstance(class_names, list):
        raise ValueError("data.yaml must define a 'names' list")

    train_images, train_labels = _resolve_split_dirs(data_cfg, "train", base_dir)
    val_images, val_labels = _resolve_split_dirs(data_cfg, "val", base_dir)
    test_images, test_labels = _resolve_split_dirs(data_cfg, "test", base_dir)

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_features, train_labels_np = _load_split_features("train", train_images, train_labels, class_names, args.max_samples)
    val_features, val_labels_np = _load_split_features("val", val_images, val_labels, class_names, args.max_samples)
    test_features, test_labels_np = _load_split_features("test", test_images, test_labels, class_names, None)

    train_split = _to_tensor_split(train_features, train_labels_np)
    val_split = _to_tensor_split(val_features, val_labels_np)
    test_split = _to_tensor_split(test_features, test_labels_np)

    train_loader, val_loader = _build_dataloaders(train_split, val_split, args.batch_size)
    test_loader = DataLoader(KeypointDataset(test_split), batch_size=args.batch_size, shuffle=False)

    model = KeypointClassifier(train_split.features.size(1), len(class_names))

    best_weights_path = output_dir / "best_classifier.pt"
    _train(model, train_loader, val_loader, args.epochs, args.learning_rate, best_weights_path)

    # Load best weights if available
    if best_weights_path.exists():
        model.load_state_dict(torch.load(best_weights_path, map_location="cpu"))

    test_acc, logits, labels = _evaluate(model, test_loader)
    preds = logits.argmax(axis=1)
    macro_f1 = f1_score(labels, preds, average="macro")
    label_indices = list(range(len(class_names)))
    report = classification_report(
        labels,
        preds,
        labels=label_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cmatrix = confusion_matrix(labels, preds, labels=label_indices).tolist()

    metrics = {
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cmatrix,
        "samples_used": {
            "train": int(train_split.labels.size(0)),
            "val": int(val_split.labels.size(0)),
            "test": int(test_split.labels.size(0)),
        },
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    _LOGGER.info("Test accuracy: %.3f | Macro F1: %.3f", test_acc, macro_f1)
    _LOGGER.info("Saved metrics to %s", metrics_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe hands baseline classifier")
    parser.add_argument("--data", type=Path, default=Path("sign_language/training/asl_letters/data.yaml"), help="Path to data.yaml")
    parser.add_argument("--output", type=Path, default=Path("sign_language/training/runs/mediapipe_baseline"), help="Destination directory for artifacts")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on samples per split for quick experiments")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

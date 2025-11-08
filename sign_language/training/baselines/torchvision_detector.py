"""Fine-tune a TorchVision detector on the ASL letters dataset."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from tqdm import tqdm

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DetectionSample:
    image: torch.Tensor
    target: Dict[str, torch.Tensor]


class ASLDataset(Dataset[DetectionSample]):
    def __init__(self, images_dir: Path, labels_dir: Path, class_names: Sequence[str]) -> None:
        self._images: List[Path] = []
        self._labels: List[Path] = []
        self._class_names = class_names

        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for image_path in sorted(images_dir.glob(pattern)):
                label_path = labels_dir / f"{image_path.stem}.txt"
                if label_path.exists():
                    self._images.append(image_path)
                    self._labels.append(label_path)

        if not self._images:
            raise RuntimeError(f"No labeled images found in {images_dir}")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_path = self._images[idx]
        label_path = self._labels[idx]

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes: List[List[float]] = []
        labels: List[int] = []

        with label_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                class_id_str, x_center_str, y_center_str, w_str, h_str = line.split()[:5]
                class_idx = int(class_id_str)
                if class_idx >= len(self._class_names):
                    continue

                x_center = float(x_center_str) * width
                y_center = float(y_center_str) * height
                box_width = float(w_str) * width
                box_height = float(h_str) * height

                xmin = max(0.0, x_center - box_width / 2)
                ymin = max(0.0, y_center - box_height / 2)
                xmax = min(width, x_center + box_width / 2)
                ymax = min(height, y_center + box_height / 2)

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_idx + 1)  # TorchVision expects 1-indexed labels

        if not boxes:
            raise RuntimeError(f"No annotations found for {image_path}")

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }

        img_tensor = F.to_tensor(image)
        return img_tensor, target


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


def collate_fn(batch: Sequence[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    images, targets = tuple(zip(*batch))
    return list(images), list(targets)


def _to_device(batch: Sequence[torch.Tensor], targets: Sequence[Dict[str, torch.Tensor]], device: torch.device):
    images = [img.to(device) for img in batch]
    targets_device: List[Dict[str, torch.Tensor]] = []
    for tgt in targets:
        tgt_device = {k: v.to(device) for k, v in tgt.items()}
        targets_device.append(tgt_device)
    return images, targets_device


def _match_predictions(
    predictions: Sequence[Dict[str, torch.Tensor]],
    targets: Sequence[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from torchvision.ops import box_iou

    per_class_scores: Dict[int, List[float]] = {c: [] for c in range(1, num_classes + 1)}
    per_class_matches: Dict[int, List[int]] = {c: [] for c in range(1, num_classes + 1)}
    gt_counts = np.zeros(num_classes + 1, dtype=np.int64)

    for preds, target in zip(predictions, targets):
        gt_boxes = target["boxes"].cpu()
        gt_labels = target["labels"].cpu()
        gt_used = torch.zeros(len(gt_boxes), dtype=torch.bool)

        for label in gt_labels:
            if label.item() <= num_classes:
                gt_counts[label.item()] += 1

        boxes = preds["boxes"].cpu()
        scores = preds["scores"].cpu()
        labels = preds["labels"].cpu()

        if boxes.numel() == 0:
            continue

        order = torch.argsort(scores, descending=True)
        boxes = boxes[order]
        scores = scores[order]
        labels = labels[order]

        if len(gt_boxes) == 0:
            for score, label in zip(scores, labels):
                per_class_scores[label.item()].append(float(score))
                per_class_matches[label.item()].append(0)
            continue

        ious = box_iou(boxes, gt_boxes)

        for idx in range(len(boxes)):
            label = labels[idx].item()
            score = float(scores[idx])
            if label > num_classes:
                continue
            best_iou, best_gt_idx = (ious[idx]).max(dim=0)
            if best_iou >= iou_threshold and not gt_used[best_gt_idx]:
                gt_used[best_gt_idx] = True
                per_class_scores[label].append(score)
                per_class_matches[label].append(1)
            else:
                per_class_scores[label].append(score)
                per_class_matches[label].append(0)

    ap_per_class = np.zeros(num_classes + 1, dtype=np.float32)
    precision_per_class: Dict[int, float] = {}
    recall_per_class: Dict[int, float] = {}
    tp_counts = np.zeros(num_classes + 1, dtype=np.int64)
    fp_counts = np.zeros(num_classes + 1, dtype=np.int64)
    fn_counts = np.zeros(num_classes + 1, dtype=np.int64)

    for cls in range(1, num_classes + 1):
        scores = np.array(per_class_scores[cls])
        matches = np.array(per_class_matches[cls])
        if scores.size == 0:
            precision_per_class[cls] = 0.0
            recall_per_class[cls] = 0.0
            ap_per_class[cls] = 0.0
            continue
        order = np.argsort(-scores)
        matches = matches[order]
        tp = np.cumsum(matches)
        fp = np.cumsum(1 - matches)
        denom = tp + fp
        precision = np.divide(tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
        recall = tp / max(gt_counts[cls], 1)

        precision_per_class[cls] = float(precision[-1]) if precision.size else 0.0
        recall_per_class[cls] = float(recall[-1]) if recall.size else 0.0

        tp_counts[cls] = int(matches.sum())
        fp_counts[cls] = int(matches.size - matches.sum())
        fn_counts[cls] = int(max(gt_counts[cls] - matches.sum(), 0))

        recall_curve = np.concatenate(([0.0], recall, [1.0]))
        precision_curve = np.concatenate(([0.0], precision, [0.0]))
        precision_curve = np.maximum.accumulate(precision_curve[::-1])[::-1]
        ap = np.trapz(precision_curve, recall_curve)
        ap_per_class[cls] = float(ap)

    precision_arr = np.array([precision_per_class.get(c, 0.0) for c in range(1, num_classes + 1)])
    recall_arr = np.array([recall_per_class.get(c, 0.0) for c in range(1, num_classes + 1)])
    tp_arr = tp_counts[1:]
    fp_arr = fp_counts[1:]
    fn_arr = fn_counts[1:]

    return ap_per_class, precision_arr, recall_arr, tp_arr, fp_arr, fn_arr


def _train_one_epoch(
    model,
    loader,
    optimiser,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(loader, desc="train"):
        images_device, targets_device = _to_device(images, targets, device)
        loss_dict = model(images_device, targets_device)
        losses = sum(loss for loss in loss_dict.values())

        optimiser.zero_grad()
        losses.backward()
        optimiser.step()

        running_loss += float(losses.item()) * len(images)

    return running_loss / len(loader.dataset)


def _evaluate_model(
    model,
    loader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, object]:
    model.eval()
    all_predictions: List[Dict[str, torch.Tensor]] = []
    all_targets: List[Dict[str, torch.Tensor]] = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="eval"):
            imgs, tgts = _to_device(images, targets, device)
            predictions = model(imgs)
            all_predictions.extend([{k: v.cpu() for k, v in pred.items()} for pred in predictions])
            all_targets.extend([{k: v.cpu() for k, v in tgt.items()} for tgt in tgts])

    ap_per_class, precision_arr, recall_arr, tp_arr, fp_arr, fn_arr = _match_predictions(all_predictions, all_targets, num_classes)
    map_50 = float(ap_per_class[1:].mean() if ap_per_class.size > 1 else 0.0)

    f1_per_class = []
    for precision, recall in zip(precision_arr, recall_arr):
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_per_class.append(float(f1))
    macro_f1 = float(np.mean(f1_per_class)) if f1_per_class else 0.0

    return {
        "map_50": map_50,
        "precision_per_class": precision_arr.tolist(),
        "recall_per_class": recall_arr.tolist(),
        "f1_per_class": f1_per_class,
        "tp_per_class": tp_arr.tolist(),
        "fp_per_class": fp_arr.tolist(),
        "fn_per_class": fn_arr.tolist(),
        "macro_f1": macro_f1,
    }


def get_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_training(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)

    data_cfg_path = Path(args.data).resolve()
    data_cfg = _load_yaml(data_cfg_path)
    base_dir = data_cfg_path.parent
    class_names = data_cfg.get("names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("data.yaml must have a non-empty 'names' list")

    train_images, train_labels = _resolve_split_dirs(data_cfg, "train", base_dir)
    val_images, val_labels = _resolve_split_dirs(data_cfg, "val", base_dir)
    test_images, test_labels = _resolve_split_dirs(data_cfg, "test", base_dir)

    train_dataset = ASLDataset(train_images, train_labels, class_names)
    val_dataset = ASLDataset(val_images, val_labels, class_names)
    test_dataset = ASLDataset(test_images, test_labels, class_names)

    device = get_device(args.device)
    pin_memory = device.type == "cuda"
    persistent = args.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    if args.model == "fasterrcnn_mobilenet_v3_large_fpn":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, len(class_names) + 1)
    elif args.model == "retinanet_resnet50_fpn":
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
            model.backbone.out_channels,
            num_anchors,
            len(class_names) + 1,
        )
    else:
        raise ValueError(f"Unsupported model {args.model}")

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.1)

    best_map = 0.0
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        loss = _train_one_epoch(model, train_loader, optimiser, device)
        metrics = _evaluate_model(model, val_loader, device, len(class_names))
        lr_scheduler.step()
        _LOGGER.info("Epoch %s/%s - loss %.4f - val mAP@0.5 %.3f", epoch, args.epochs, loss, metrics["map_50"])

        if metrics["map_50"] > best_map:
            best_map = metrics["map_50"]
            torch.save(model.state_dict(), weights_path)

    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))

    test_metrics = _evaluate_model(model, test_loader, device, len(class_names))
    _LOGGER.info("Test mAP@0.5: %.3f | Macro F1: %.3f", test_metrics["map_50"], test_metrics["macro_f1"])

    metrics = {
        "val_best_map_50": best_map,
        "test_map_50": test_metrics["map_50"],
        "test_macro_f1": test_metrics["macro_f1"],
        "precision_per_class": test_metrics["precision_per_class"],
        "recall_per_class": test_metrics["recall_per_class"],
        "f1_per_class": test_metrics["f1_per_class"],
        "tp_per_class": test_metrics["tp_per_class"],
        "fp_per_class": test_metrics["fp_per_class"],
        "fn_per_class": test_metrics["fn_per_class"],
        "class_names": class_names,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    _LOGGER.info("Saved metrics to %s", metrics_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TorchVision detector baseline")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("sign_language/training/asl_letters_v2/data.yaml"),
        help="Path to data.yaml",
    )
    parser.add_argument("--output", type=Path, default=Path("sign_language/training/runs/torchvision_detector"), help="Directory for outputs")
    parser.add_argument("--model", type=str, default="fasterrcnn_mobilenet_v3_large_fpn", choices=[
        "fasterrcnn_mobilenet_v3_large_fpn",
        "retinanet_resnet50_fpn",
    ])
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--device", type=str, default=None, help="Torch device id (cpu, cuda, mps)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (0 for main process only)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

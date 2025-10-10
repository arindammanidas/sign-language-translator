"""Utility to convert the ASL dataset into TFRecords for TF Object Detection API."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import tensorflow as tf
import yaml
from PIL import Image

_LOGGER = logging.getLogger(__name__)


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_images_labels(base_dir: Path, split_key: str, cfg: Dict[str, object]) -> tuple[Path, Path]:
    split_path = cfg.get(split_key)
    if not isinstance(split_path, str):
        raise ValueError(f"Missing '{split_key}' entry in data.yaml")
    images_dir = (base_dir / split_path).resolve()
    labels_dir = images_dir.parent / "labels"
    return images_dir, labels_dir


def _iterate_labels(images_dir: Path, labels_dir: Path) -> Iterable[tuple[Path, Path]]:
    candidates: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        candidates.extend(images_dir.glob(pattern))
    for image_path in sorted(candidates):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            yield image_path, label_path


def _load_image_dimensions(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def _create_tf_example(
    image_path: Path,
    label_path: Path,
    class_names: Sequence[str],
) -> tf.train.Example | None:
    width, height = _load_image_dimensions(image_path)
    with tf.io.gfile.GFile(str(image_path), "rb") as fid:
        encoded = fid.read()

    xmins: List[float] = []
    xmaxs: List[float] = []
    ymins: List[float] = []
    ymaxs: List[float] = []
    classes_text: List[bytes] = []
    classes: List[int] = []

    with label_path.open("r", encoding="utf-8") as labels_file:
        for line in labels_file:
            if not line.strip():
                continue
            class_id_str, x_center_str, y_center_str, width_str, height_str = line.split()[:5]
            class_id = int(class_id_str)
            if class_id >= len(class_names):
                continue
            x_center = float(x_center_str)
            y_center = float(y_center_str)
            box_width = float(width_str)
            box_height = float(height_str)

            xmin = x_center - box_width / 2
            xmax = x_center + box_width / 2
            ymin = y_center - box_height / 2
            ymax = y_center + box_height / 2

            xmins.append(max(0.0, xmin))
            xmaxs.append(min(1.0, xmax))
            ymins.append(max(0.0, ymin))
            ymaxs.append(min(1.0, ymax))
            classes.append(class_id + 1)  # TFOD expects 1-indexed class ids
            classes_text.append(class_names[class_id].encode("utf-8"))

    if not xmins:
        return None

    feature = {
        "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        "image/filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path.name.encode("utf-8")])),
        "image/source_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path.name.encode("utf-8")])),
        "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded])),
        "image/format": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path.suffix.replace(".", "").encode("utf-8")])),
        "image/object/bbox/xmin": tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        "image/object/bbox/xmax": tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        "image/object/bbox/ymin": tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        "image/object/bbox/ymax": tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        "image/object/class/text": tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        "image/object/class/label": tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def _write_tfrecord(output_path: Path, examples: Iterable[tf.train.Example]) -> int:
    count = 0
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for example in examples:
            writer.write(example.SerializeToString())
            count += 1
    return count


def _export_split(
    name: str,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    class_names: Sequence[str],
) -> None:
    tf_examples = []

    for image_path, label_path in _iterate_labels(images_dir, labels_dir):
        example = _create_tf_example(image_path, label_path, class_names)
        if example is None:
            continue
        tf_examples.append(example)

    record_path = output_dir / f"{name}.record"
    count = _write_tfrecord(record_path, tf_examples)
    _LOGGER.info("Wrote %s (%s examples)", record_path, count)


def _write_label_map(output_path: Path, class_names: Sequence[str]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for index, name in enumerate(class_names, start=1):
            handle.write("item {\n")
            handle.write(f"  id: {index}\n")
            handle.write(f"  name: '{name}'\n")
            handle.write("}\n\n")
    _LOGGER.info("Label map saved to %s", output_path)


def run_export(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)

    data_cfg_path = Path(args.data).resolve()
    data_cfg = _load_yaml(data_cfg_path)
    base_dir = data_cfg_path.parent
    class_names = data_cfg.get("names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("data.yaml must provide a non-empty 'names' list")

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        images_dir, labels_dir = _resolve_images_labels(base_dir, split, data_cfg)
        _export_split(split, images_dir, labels_dir, output_dir, class_names)

    _write_label_map(output_dir / "label_map.pbtxt", class_names)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dataset to TFRecord for TensorFlow Object Detection API")
    parser.add_argument("--data", type=Path, default=Path("sign_language/training/asl_letters/data.yaml"), help="Path to data.yaml")
    parser.add_argument("--output", type=Path, default=Path("sign_language/training/runs/tfod"), help="Output directory for TFRecords")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_export(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

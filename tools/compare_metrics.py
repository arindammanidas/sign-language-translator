"""Compare YOLO, MediaPipe, and TorchVision metrics files."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:  # Optional plotting
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class MetricsSpec:
    name: str
    path: Path


DEFAULT_SPECS: Sequence[MetricsSpec] = (
    MetricsSpec("YOLO11", Path("sign_language/training/runs/yolo_eval/metrics.json")),
    MetricsSpec("MediaPipe + MLP", Path("sign_language/training/runs/mediapipe_baseline/metrics.json")),
    MetricsSpec("TorchVision", Path("sign_language/training/runs/torchvision_detector/metrics.json")),
)


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    raise ValueError(f"Unsupported metrics format in {path}")


def _mean(values: Sequence[float]) -> float:
    arr = [v for v in values if v is not None]
    return float(sum(arr) / len(arr)) if arr else float("nan")


def _extract_yolo(metrics: Dict[str, object]) -> Dict[str, float]:
    precision = metrics.get("precision_per_class")
    recall = metrics.get("recall_per_class")

    precision_avg = float(precision) if isinstance(precision, (int, float)) else _mean(list(precision or []))
    recall_avg = float(recall) if isinstance(recall, (int, float)) else _mean(list(recall or []))
    if precision_avg + recall_avg > 0:
        macro_f1 = 2 * precision_avg * recall_avg / (precision_avg + recall_avg)
    else:
        macro_f1 = float(metrics.get("macro_f1", float("nan")))

    return {
        "score": float(metrics.get("map_50", float("nan"))),
        "precision_avg": precision_avg,
        "recall_avg": recall_avg,
        "macro_f1": macro_f1,
    }


def _extract_mediapipe(metrics: Dict[str, object]) -> Dict[str, float]:
    report = metrics.get("classification_report", {})
    precision_vals = []
    recall_vals = []
    f1_vals = []
    if isinstance(report, dict):
        for value in report.values():
            if isinstance(value, dict):
                precision_vals.append(value.get("precision"))
                recall_vals.append(value.get("recall"))
                f1_vals.append(value.get("f1-score"))

    return {
        "score": float(metrics.get("test_accuracy", float("nan"))),
        "precision_avg": _mean(precision_vals),
        "recall_avg": _mean(recall_vals),
        "macro_f1": float(metrics.get("macro_f1", _mean(f1_vals))),
    }


def _extract_torchvision(metrics: Dict[str, object]) -> Dict[str, float]:
    tp = metrics.get("tp_per_class", []) or []
    fp = metrics.get("fp_per_class", []) or []
    fn = metrics.get("fn_per_class", []) or []

    tp = [float(v) for v in tp]
    fp = [float(v) for v in fp]
    fn = [float(v) for v in fn]

    precision_vals: List[float] = []
    recall_vals: List[float] = []
    f1_vals: List[float] = []
    for t, f, n in zip(tp, fp, fn):
        if t + f > 0:
            precision = t / (t + f)
        else:
            precision = 0.0
        if t + n > 0:
            recall = t / (t + n)
        else:
            recall = 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        precision_vals.append(precision)
        recall_vals.append(recall)
        f1_vals.append(f1)

    return {
        "score": float(metrics.get("test_map_50", float("nan"))),
        "precision_avg": _mean(precision_vals),
        "recall_avg": _mean(recall_vals),
        "macro_f1": float(metrics.get("test_macro_f1", _mean(f1_vals))),
    }


def collect_rows(specs: Sequence[MetricsSpec]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    missing: List[str] = []

    for spec in specs:
        if not spec.path.exists():
            missing.append(f"{spec.name} ({spec.path})")
            continue

        metrics = _load_json(spec.path)
        name_lower = spec.name.lower()
        if name_lower.startswith("yolo"):
            extracted = _extract_yolo(metrics)
        elif "mediapipe" in name_lower:
            extracted = _extract_mediapipe(metrics)
        else:
            extracted = _extract_torchvision(metrics)

        rows.append(
            {
                "Model": spec.name,
                "mAP@0.5/Accuracy": extracted["score"],
                "Macro F1": extracted["macro_f1"],
                "Precision (avg)": extracted["precision_avg"],
                "Recall (avg)": extracted["recall_avg"],
            }
        )

    if missing:
        _LOGGER.warning("Missing metrics: %s", ", ".join(missing))
    if not rows:
        raise RuntimeError("No metrics files found")

    rows.sort(key=lambda r: (math.isnan(r["mAP@0.5/Accuracy"]), -r["mAP@0.5/Accuracy"]))
    return rows


def markdown_table(rows: Sequence[Dict[str, float]]) -> str:
    headers = list(rows[0].keys())
    widths = [len(header) for header in headers]

    formatted_rows: List[List[str]] = []
    for row in rows:
        formatted_row = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                cell = "nan" if math.isnan(value) else f"{value:.6f}".rstrip("0").rstrip(".")
            else:
                cell = str(value)
            widths[headers.index(header)] = max(widths[headers.index(header)], len(cell))
            formatted_row.append(cell)
        formatted_rows.append(formatted_row)

    def format_line(row_values: Sequence[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row_values)) + " |"

    header_line = format_line(headers)
    separator_line = "|:" + ":|:".join("-" * widths[idx] for idx in range(len(headers))) + ":|"
    data_lines = [format_line(row) for row in formatted_rows]

    return "\n".join([header_line, separator_line] + data_lines) + "\n"


def maybe_save_plot(rows: Sequence[Dict[str, float]], output_path: Path) -> None:
    if plt is None:
        _LOGGER.warning("matplotlib not available; skipping plot generation")
        return

    labels = [row["Model"] for row in rows]
    scores = [row["mAP@0.5/Accuracy"] for row in rows]
    macro_f1 = [row["Macro F1"] for row in rows]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], scores, width, label="mAP@0.5/Accuracy")
    ax.bar([i + width / 2 for i in x], macro_f1, width, label="Macro F1")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    _LOGGER.info("Plot saved to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model metrics JSON files")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional entries of the form ModelName=path/to/metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sign_language/training/runs/comparison"),
        help="Directory for generated artifacts",
    )
    return parser.parse_args()


def parse_specs(args: Sequence[str]) -> Sequence[MetricsSpec]:
    if not args:
        return DEFAULT_SPECS
    specs: List[MetricsSpec] = []
    for entry in args:
        if "=" not in entry:
            raise ValueError(f"Invalid metrics specification '{entry}'. Use name=path format.")
        name, path_str = entry.split("=", 1)
        specs.append(MetricsSpec(name.strip(), Path(path_str.strip()).resolve()))
    return specs


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    specs = parse_specs(args.metrics)
    rows = collect_rows(specs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "comparison_table.md"
    plot_path = args.output_dir / "comparison_plot.png"

    table_md = markdown_table(rows)
    table_path.write_text(table_md, encoding="utf-8")
    _LOGGER.info("Comparison table saved to %s", table_path)
    print(table_md)

    maybe_save_plot(rows, plot_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

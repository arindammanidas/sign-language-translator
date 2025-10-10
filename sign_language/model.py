from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Settings

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class BoundingBox:
    """Axis-aligned bounding box normalised to [0, 1] range."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(slots=True)
class DetectionResult:
    """Simple container for a single detection result."""

    label: str
    confidence: float
    bounding_box: Optional[BoundingBox]


class YOLOSignLanguageModel:
    """Wrapper around a YOLO model specialised for ASL detection."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = self._load_model(settings.model_path)
        self._confidence_threshold = settings.confidence_threshold

    @property
    def ready(self) -> bool:
        return self._model is not None

    @staticmethod
    def _load_model(path: Path):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover - executed only when YOLO missing
            raise RuntimeError(
                "ultralytics package is required. Install dependencies via requirements.txt."
            ) from exc

        if not path.exists():
            _LOGGER.warning(
                "YOLO weights not found at %s. The service will run without predictions until a model is supplied.",
                path,
            )
            return None

        try:
            return YOLO(str(path))
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to load YOLO weights from {path}: {exc}") from exc

    def predict(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """Run inference on a single frame and return the strongest detection."""

        if not self.ready:
            return None

        results = self._model.predict(frame, conf=self._confidence_threshold, verbose=False)
        if not results:
            return None

        result = results[0]

        if result.probs is not None:
            top_idx = int(result.probs.top1)
            if top_idx < 0:
                return None
            label = result.names[top_idx]
            confidence = float(result.probs.top1conf)
            return DetectionResult(label=label, confidence=confidence, bounding_box=None)

        if result.boxes is None or result.boxes.cls is None or not len(result.boxes):
            return None

        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        if len(scores) == 0:
            return None

        max_idx = int(scores.argmax())
        confidence = float(scores[max_idx])
        label = result.names[int(classes[max_idx])]
        if confidence < self._confidence_threshold:
            return None

        bbox = None
        if result.boxes is not None and result.boxes.xyxyn is not None:
            try:
                coords = result.boxes.xyxyn[max_idx].cpu().numpy().tolist()
            except Exception:  # pragma: no cover - fallback if tensor conversion fails
                coords = None
            if coords and len(coords) >= 4:
                x1, y1, x2, y2 = (float(max(0.0, min(1.0, value))) for value in coords[:4])
                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

        return DetectionResult(label=label, confidence=confidence, bounding_box=bbox)

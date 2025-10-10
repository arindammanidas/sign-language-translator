from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional

from .model import BoundingBox, DetectionResult


@dataclass
class SmoothedPrediction:
    label: str
    confidence: float
    bounding_box: Optional[BoundingBox]


class PredictionSmoother:
    """Maintain a moving window of predictions to mitigate jitter."""

    def __init__(self, window_size: int) -> None:
        self._window_size = max(1, window_size)
        self._buffer: Deque[DetectionResult] = deque(maxlen=self._window_size)

    def update(self, detection: Optional[DetectionResult]) -> Optional[SmoothedPrediction]:
        if detection is None:
            self._buffer.clear()
            return None

        self._buffer.append(detection)
        if not self._buffer:
            return None

        label_counts = Counter(item.label for item in self._buffer)
        label, _ = label_counts.most_common(1)[0]
        confidences = [item.confidence for item in self._buffer if item.label == label]
        if not confidences:
            return None

        averaged_confidence = sum(confidences) / len(confidences)
        bounding_box = None
        for item in reversed(self._buffer):
            if item.label == label:
                bounding_box = item.bounding_box
                if bounding_box is not None:
                    break

        return SmoothedPrediction(
            label=label,
            confidence=averaged_confidence,
            bounding_box=bounding_box,
        )

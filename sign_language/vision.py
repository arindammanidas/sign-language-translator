from __future__ import annotations

import base64
import binascii
from typing import Optional

import cv2
import numpy as np


def decode_image(data_url: str) -> Optional[np.ndarray]:
    """Convert a data URL from the browser into a BGR numpy array."""

    if not data_url:
        return None

    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    try:
        binary = base64.b64decode(encoded)
    except (ValueError, binascii.Error):
        return None

    array = np.frombuffer(binary, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return frame

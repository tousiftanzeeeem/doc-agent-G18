"""Stage 1 — deskew / denoise / binarize / augment"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import Page
from ..logging_conf import get_logger
from .loader import load_image, save_image

log = get_logger("ingest.preprocess")


def _skew_angle(bin_img: np.ndarray) -> float:
    """Estimate page skew (degrees) from the text's min-area rect (downscaled copy)."""
    import cv2

    small = cv2.resize(bin_img, (bin_img.shape[1] // 4, bin_img.shape[0] // 4))
    coords = np.column_stack(np.where(small > 0))
    if len(coords) < 100:
        return 0.0
    (_, _), (w, h), angle = cv2.minAreaRect(coords.astype(np.float32))
    angle = float(angle)
    # cv2 returns angle in [-90, 0); normalize to the small tilt we care about
    if w < h:
        angle = 90.0 + angle
    if abs(angle) > 45.0:
        angle = 0.0
    return angle
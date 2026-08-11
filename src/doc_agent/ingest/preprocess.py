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


def _preprocess_one(page: Page, cfg: dict) -> Page:
    import cv2

    img = load_image(page)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if cfg.get("denoise", True):
        # fast denoise: Gaussian blur (heavy NL-means is redundant with the enhance stage)
        gray = cv2.GaussianBlur(gray, (3, 3), 0.8)

    if cfg.get("binarize", True):
        # Otsu inverse binarization: ink = white, paper = black (contour-friendly)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        bin_img = gray

    if cfg.get("deskew", True):
        angle = _skew_angle(bin_img)
        if abs(angle) > 0.5:
            h, w = bin_img.shape
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            bin_img = cv2.warpAffine(
                bin_img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
            )
            gray = cv2.warpAffine(
                gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
            )

    # keep grayscale for OCR/layout; the binarized mask is derivable on demand
    out_dir = Path(str(cfg.get("out_dir", "data/interim/processed")))
    return save_image(page, gray, out_dir)    
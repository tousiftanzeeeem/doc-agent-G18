"""Stage 2 — layout detection / segmentation

Classical projection/contour approach (chosen over a detection model — see design_choices.md):
binarize -> morphological close merges text lines into blocks -> connected components -> merge
overlapping blocks -> classify each block (text | table | figure | heading).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..contracts import Page, Region
from ..ingest.loader import load_image
from ..logging_conf import get_logger

log = get_logger("vision.layout")


def _blocks(gray: np.ndarray, cfg: dict) -> list[tuple[int, int, int, int]]:
    """Return (x, y, w, h) ink blocks via morphology + connected components."""
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = bin_img.shape
    min_h = int(cfg.get("min_block_h", 20))
    merge_gap = int(cfg.get("merge_gap", 14))
    kx = max(15, w // 60)
    ky = max(9, h // 90)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] * b[3] >= min_h * min_h]

    # merge boxes with strong overlap
    merged: list[list[int]] = []
    for x, y, w, h in sorted(boxes, key=lambda b: (b[1], b[0])):
        placed = False
        for m in merged:
            mx, my, mw, mh = m
            ix = max(0, min(x + w, mx + mw) - max(x, mx))
            iy = max(0, min(y + h, my + mh) - max(y, my))
            overlap = ix * iy
            if overlap > 0.5 * min(w * h, mw * mh) or (
                iy > 0.6 * min(h, mh) and abs(x - mx) < merge_gap
            ):
                nx = min(x, mx)
                ny = min(y, my)
                m[:] = [nx, ny, max(x + w, mx + mw) - nx, max(y + h, my + mh) - ny]
                placed = True
                break
        if not placed:
            merged.append([x, y, w, h])
    return [(m[0], m[1], m[2], m[3]) for m in merged]


def _has_ruling_lines(gray: np.ndarray, box: tuple[int, int, int, int], img_w: int) -> bool:
    """Tables show up as runs of long horizontal rules; Hough detects them cheaply."""
    x, y, w, h = box
    if w < img_w // 4:
        return False
    _, b = cv2.threshold(
        gray[y : y + h, x : x + w], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    lines = cv2.HoughLinesP(
        b, 1, np.pi / 180, threshold=max(20, w // 15), minLineLength=int(0.55 * w)
    )
    return lines is not None and len(lines) >= 3


def _classify(gray: np.ndarray, box: tuple[int, int, int, int], img_h: int, img_w: int) -> str:
    x, y, w, h = box
    patch = gray[y : y + h, x : x + w]
    density = float((patch < 128).mean())

    if _has_ruling_lines(gray, box, img_w):
        return "table"

    page_area = img_h * img_w
    if h * w > 0.20 * page_area and (density < 0.03 or density > 0.45):
        return "figure"  # plates/halftones: nearly blank or nearly solid ink

    # running titles, section heads: short, wide, sitting at the page edge
    if h <= 1.7 * max(12, img_h // 110) and (y < 0.12 * img_h or y + h > 0.92 * img_h):
        return "heading"
    return "text"


def _reading_order(
    boxes: list[tuple[int, int, int, int]], img_h: int
) -> list[tuple[int, int, int, int]]:
    """Banded sort: group blocks into rows of ~median height, sort each row left->right."""
    if not boxes:
        return boxes
    med_h = float(np.median([b[3] for b in boxes]))
    bands: dict[int, list[tuple[int, int, int, int]]] = {}
    for b in boxes:
        band = int(b[1] // max(med_h, 1))
        bands.setdefault(band, []).append(b)
    ordered: list[tuple[int, int, int, int]] = []
    for band in sorted(bands):
        ordered.extend(sorted(bands[band], key=lambda b: (b[0], b[1])))
    return ordered

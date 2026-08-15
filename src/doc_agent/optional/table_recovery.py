"""Stage 2.5 — table-structure recovery for UNRULED whitespace-aligned tables.

Core A1 contribution.  Projection-profile column/row segmentation:
  1. Compute vertical ink projection → find deep whitespace valleys → column boundaries.
  2. Compute horizontal ink projection → find row gaps → row boundaries.
  3. Intersect column and row boundaries → grid of cell bboxes.

Designed for the 1889 Bengal sanitary-report tables that have NO ruling lines —
just whitespace-aligned columns and multi-row headers.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..logging_conf import get_logger

log = get_logger("vision.table_recovery")


# ---------------------------------------------------------------------------
# Projection-profile helpers
# ---------------------------------------------------------------------------

def _vertical_projection(bin_img: np.ndarray) -> np.ndarray:
    """Count ink pixels per column (vertical projection profile)."""
    return np.sum(bin_img > 0, axis=0).astype(np.float64)


def _horizontal_projection(bin_img: np.ndarray) -> np.ndarray:
    """Count ink pixels per row (horizontal projection profile)."""
    return np.sum(bin_img > 0, axis=1).astype(np.float64)


def _find_gaps(profile: np.ndarray, min_gap: int, threshold_ratio: float = 0.05) -> list[int]:
    """Find boundaries where the projection profile drops below *threshold_ratio*
    of its maximum for at least *min_gap* consecutive pixels.

    Gaps touching either edge of the profile are ignored: a blank margin is paper,
    not a column/row separator (otherwise prose blocks look like 2-column tables
    and tables gain phantom edge rows).

    Returns the midpoint of each internal gap (sorted).
    """
    if profile.max() == 0:
        return []
    thresh = profile.max() * threshold_ratio
    below = profile < thresh  # True where ink is sparse

    gaps: list[tuple[int, int]] = []
    start = None
    for i, b in enumerate(below):
        if b and start is None:
            start = i
        elif not b and start is not None:
            # internal gap only: must have ink on BOTH sides (start > 0, i < len)
            if i - start >= min_gap and start > 0 and i < len(profile):
                gaps.append((start, i))
            start = None
    # trailing run of blank: touches the right edge -> a margin, never a separator

    return [int((s + e) // 2) for s, e in gaps]


def _boundaries_from_gaps(gaps: list[int], total_len: int) -> list[tuple[int, int]]:
    """Convert gap midpoints into (start, end) spans for columns or rows."""
    edges = [0] + gaps + [total_len]
    spans: list[tuple[int, int]] = []
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i + 1]
        if e - s > 2:  # skip degenerate spans
            spans.append((s, e))
    return spans


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recover_cells(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    min_col_gap: int = 8,
    min_row_gap: int = 4,
) -> list[list[tuple[int, int, int, int]]]:
    """Recover a 2-D cell grid from a table region (no ruling lines required).

    Parameters
    ----------
    gray : np.ndarray
        Full-page grayscale image.
    bbox : tuple
        (x1, y1, x2, y2) bounding box of the table region.
    min_col_gap : int
        Minimum whitespace gap width (px) to count as a column separator.
    min_row_gap : int
        Minimum whitespace gap height (px) to count as a row separator.

    Returns
    -------
    list[list[tuple]]
        Row-major 2-D list of cell bboxes ``(x1, y1, x2, y2)`` in **page
        coordinates** (not crop coordinates).
    """
    x1, y1, x2, y2 = bbox
    crop = gray[y1:y2, x1:x2]

    # Binarize the crop (ink = white)
    _, bin_crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- column boundaries (vertical projection) ---
    v_proj = _vertical_projection(bin_crop)
    col_gaps = _find_gaps(v_proj, min_gap=min_col_gap)
    col_spans = _boundaries_from_gaps(col_gaps, bin_crop.shape[1])

    # --- row boundaries (horizontal projection) ---
    h_proj = _horizontal_projection(bin_crop)
    row_gaps = _find_gaps(h_proj, min_gap=min_row_gap)
    row_spans = _boundaries_from_gaps(row_gaps, bin_crop.shape[0])

    if len(col_spans) < 2:
        log.debug("table_recovery: only %d column(s) detected — treating as single-column text",
                  len(col_spans))

    # --- build grid (convert back to page coordinates) ---
    grid: list[list[tuple[int, int, int, int]]] = []
    for rs, re in row_spans:
        row: list[tuple[int, int, int, int]] = []
        for cs, ce in col_spans:
            row.append((x1 + cs, y1 + rs, x1 + ce, y1 + re))
        grid.append(row)

    log.debug("table_recovery: %d rows × %d cols from bbox %s",
              len(row_spans), len(col_spans), bbox)
    return grid


def has_whitespace_columns(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    min_col_gap: int = 8,
    min_cols: int = 2,
) -> bool:
    """Quick check: does this region contain ≥ *min_cols* whitespace-separated columns?

    Used by ``layout._classify`` to detect unruled tables without running the
    full cell-recovery pass.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    if w < 80:  # too narrow to be a real table
        return False

    crop = gray[y1:y2, x1:x2]
    _, bin_crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    v_proj = _vertical_projection(bin_crop)
    col_gaps = _find_gaps(v_proj, min_gap=min_col_gap)
    col_spans = _boundaries_from_gaps(col_gaps, bin_crop.shape[1])

    return len(col_spans) >= min_cols

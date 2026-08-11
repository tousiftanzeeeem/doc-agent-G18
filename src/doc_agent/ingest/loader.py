"""Stage 1 — load scanned page-images"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..contracts import Page
from ..logging_conf import get_logger

log = get_logger("ingest.loader")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

def _render_pdf(pdf_path: Path, out_dir: Path, dpi: int = 200) -> list[Path]:
    """Render every page of a scanned PDF to JPEGs (pypdfium2, no poppler needed)."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0
    rendered: list[Path] = []
    for i in range(len(doc)):
        out = out_dir / f"page_{i + 1:04d}.jpg"
        if out.exists():
            rendered.append(out)
            continue
        img = doc[i].render(scale=scale).to_pil()
        img.convert("RGB").save(out, quality=88)
        rendered.append(out)
    return rendered
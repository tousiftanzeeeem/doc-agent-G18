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


def _discover_pages(root: Path, cfg: dict) -> list[Page]:
    """PDFs are rendered to page images once; then every page image becomes a Page."""
    dpi = int(cfg.get("dpi", 200))
    pages: list[Page] = []
    seen: set[Path] = set()

    # 1) render any scanned PDFs found directly under root
    for pdf in sorted(root.glob("*.pdf")):
        doc_id = pdf.stem
        out_dir = root / "pages" / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        rendered = _render_pdf(pdf, out_dir, dpi)
        for i, img in enumerate(sorted(rendered)):
            pages.append(Page(id=f"{doc_id}_p{i + 1:04d}", image_path=str(img), doc_id=doc_id))
            seen.add(img)

    # 2) any page images already sitting in root/pages/**
    for img in sorted(root.rglob("*")):
        if img.is_file() and img.suffix.lower() in _IMAGE_EXTS and img not in seen:
            doc_id = img.parent.name
            m = re.search(r"(\d+)(?!.*\d)", img.stem) or re.search(r"\d+", img.stem)
            idx = int(m.group(0)) if m else 0
            pages.append(Page(id=f"{doc_id}_p{idx:04d}", image_path=str(img), doc_id=doc_id))

    # deterministic order: doc, then page index
    pages.sort(key=lambda p: (p.doc_id, p.id))
    return pages
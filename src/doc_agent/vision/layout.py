"""Stage 2 — layout detection / segmentation"""
from __future__ import annotations
from ..contracts import *  # noqa

def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect text/table/figure/heading regions. IMPLEMENT."""
    raise NotImplementedError("Stage 2: layout detection")


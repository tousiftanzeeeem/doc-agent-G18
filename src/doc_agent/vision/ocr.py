"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)"""
from __future__ import annotations
from ..contracts import *  # noqa

class Reader:
    """Model set by cfg['ocr']. Baseline: pretrained TrOCR/Donut/Tesseract."""
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]
    def transcribe_region(self, region: Region) -> str:
        raise NotImplementedError("Stage 3: transcribe a region")

def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Regions -> text chunks. IMPLEMENT (calls Reader)."""
    raise NotImplementedError("Stage 3: OCR to chunks")


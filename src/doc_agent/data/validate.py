"""Data — data schema/quality validation at ingest"""
from __future__ import annotations
from ..contracts import *  # noqa

def validate(pages: list[Page]) -> None:
    """Assert min pages/words, format, no leakage across splits. IMPLEMENT."""
    raise NotImplementedError("Data: validate")


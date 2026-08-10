"""Data — corpus versioning (which corpus version -> which result)"""
from __future__ import annotations
from ..contracts import *  # noqa

def snapshot(corpus_dir: str) -> str:
    """Hash + record a corpus version id. IMPLEMENT (or wire DVC)."""
    raise NotImplementedError("Data: version snapshot")


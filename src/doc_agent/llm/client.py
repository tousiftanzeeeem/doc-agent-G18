"""LLM — the single LLM call wrapper (all model calls go through here)"""
from __future__ import annotations
from ..contracts import *  # noqa

class LLM:
    """Model set by cfg['agent']. Key from settings. IMPLEMENT complete()."""
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
    def complete(self, prompt: str, **kw) -> str:
        raise NotImplementedError("LLM: complete")


"""Stage 1 — ENHANCEMENT (VAE / diffusion) — generative denoise / super-resolution of degraded scans"""
from __future__ import annotations
from ..contracts import *  # noqa

class Enhancer:
    """Model set by cfg['enhance']. IMPLEMENT train() and apply()."""
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["enhance"]
    def train(self, pages: list[Page]) -> None:
        raise NotImplementedError("Stage 1: train VAE/diffusion enhancer")
    def apply(self, pages: list[Page]) -> list[Page]:
        raise NotImplementedError("Stage 1: apply enhancer")

def run(pages: list[Page], cfg: dict) -> list[Page]:
    if not cfg["enhance"]["enabled"]:
        return pages
    return Enhancer(cfg).apply(pages)


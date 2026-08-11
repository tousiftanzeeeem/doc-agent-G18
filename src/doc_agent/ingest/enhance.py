"""Stage 1 — ENHANCEMENT (VAE / diffusion) — generative denoise / super-resolution of degraded scans"""

import numpy as np

from __future__ import annotations
from ..contracts import *  # noqa
from .loader import load_image, save_image

# class Enhancer:
#     """Model set by cfg['enhance']. IMPLEMENT train() and apply()."""
#     def __init__(self, cfg: dict) -> None:
#         self.cfg = cfg["enhance"]
#     def train(self, pages: list[Page]) -> None:
#         raise NotImplementedError("Stage 1: train VAE/diffusion enhancer")
#     def apply(self, pages: list[Page]) -> list[Page]:
#         raise NotImplementedError("Stage 1: apply enhancer")


class Enhancer:
    """Model set by cfg['enhance']; train() learns the denoiser, apply() enhances pages."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["enhance"]
        self.device = "cpu"


def _collect_patches(
    pages: list[Page], rng: np.random.Generator, patch: int, stride: int, max_patches: int
) -> list[np.ndarray]:
    """Extract square clean patches from a few pages (deterministic stride sampling)."""
    import cv2

    from .preprocess import augment

    patches: list[np.ndarray] = []
    for p in pages[: min(len(pages), 12)]:
        img = load_image(p)
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (512, 700))
        for y in range(0, g.shape[0] - patch + 1, stride):
            for x in range(0, g.shape[1] - patch + 1, stride):
                patches.append(augment(g[y : y + patch, x : x + patch], rng))
                if len(patches) >= max_patches:
                    return patches
    return patches


def run(pages: list[Page], cfg: dict) -> list[Page]:
    if not cfg["enhance"]["enabled"]:
        return pages
    return Enhancer(cfg).apply(pages)

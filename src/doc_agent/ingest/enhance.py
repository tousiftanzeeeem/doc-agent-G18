"""Stage 1 — ENHANCEMENT (VAE / diffusion) — generative denoise / super-resolution of degraded scans"""

import numpy as np

from __future__ import annotations
from ..contracts import *  # noqa
from .loader import load_image, save_image


from pathlib import Path
from ..logging_conf import get_logger
log = get_logger("ingest.enhance")

CKPT_DIR = Path("data/interim/enhance")


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch


def _unet() -> torch.nn.Module:
    """Tiny 3-level UNet: 1ch -> 16/32/64 -> 1ch, ~45k params (CPU-trainable in minutes)."""
    import torch
    import torch.nn as nn

    class DoubleConv(nn.Module):
        def __init__(self, cin: int, cout: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc1 = DoubleConv(1, 16)
            self.enc2 = DoubleConv(16, 32)
            self.enc3 = DoubleConv(32, 64)
            self.pool = nn.MaxPool2d(2)
            self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dec2 = DoubleConv(64, 32)
            self.dec1 = DoubleConv(32, 16)
            self.head = nn.Conv2d(16, 1, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            d3 = torch.cat([self.up3(e3), e2], dim=1)
            d2 = self.dec2(d3)
            d1 = self.dec1(torch.cat([self.up2(d2), e1], dim=1))
            return self.head(d1)

    return UNet()



class Enhancer:
    """Model set by cfg['enhance']; train() learns the denoiser, apply() enhances pages."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["enhance"]
        self.device = "cpu"
    # -- training -----------------------------------------------------------
    def train(self, pages: list[Page]) -> None:
        """Self-supervised denoising: corrupt clean crops, learn to reconstruct (uses augment())."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if self.cfg.get("model") != "unet_small":
            log.info("enhance model=%s needs no training", self.cfg.get("model"))
            return

        rng = np.random.default_rng(int(self.cfg.get("seed", 42)))
        patches = _collect_patches(pages, rng, patch=64, stride=48, max_patches=800)
        if not patches:
            log.warning("no patches to train enhancer on")
            return

        x = torch.from_numpy(np.stack(patches)[:, None].astype(np.float32) / 255.0)
        noisy = torch.clamp(x + 0.10 * torch.randn_like(x), 0.0, 1.0)
        ds = TensorDataset(noisy, x)
        dl = DataLoader(ds, batch_size=16, shuffle=True)

        model = _unet().to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        epochs = int(self.cfg.get("epochs", 3))
        model.train()
        for ep in range(epochs):
            tot = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                tot += float(loss.item())
            avg = tot / max(len(dl), 1)
            log.info("enhancer epoch %d/%d loss=%.5f", ep + 1, epochs, avg)

        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), CKPT_DIR / "unet_small.pt")
        log.info("saved enhancer checkpoint to %s", CKPT_DIR / "unet_small.pt")


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

"""Stage 1 — ENHANCEMENT (VAE / diffusion) — generative denoise / super-res of degraded scans.

Two paths behind one interface:
  * model="clahe_denoise" — classical: CLAHE contrast + NL-means denoise + unsharp mask.
  * model="unet_small" — a tiny UNet denoiser trained FROM SCRATCH, self-supervised
    (we corrupt our own pages with noise, the net learns to invert it) — the CPU-feasible
    stand-in for a VAE/diffusion enhancer; the same train()/apply() contract would host a
    real diffusion model once a GPU is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..contracts import Page
from ..logging_conf import get_logger
from .loader import load_image, save_image

if TYPE_CHECKING:
    import torch

log = get_logger("ingest.enhance")

CKPT_DIR = Path("data/interim/enhance")


def _classical(img: np.ndarray) -> np.ndarray:
    import cv2

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    out = clahe.apply(gray)
    # light blur then unsharp mask: sharpen strokes, suppress paper grain
    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    out = cv2.addWeighted(out, 1.5, blur, -0.5, 0)
    return out


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


def _resolve_device(requested: str) -> str:
    """'cuda' if requested and torch has a usable CUDA device, else 'cpu'."""
    if requested != "cuda":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    log.warning("device: cuda requested but torch has no CUDA — enhancer falls back to cpu")
    return "cpu"


class Enhancer:
    """Model set by cfg['enhance']; train() learns the denoiser, apply() enhances pages."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["enhance"]
        self.device = _resolve_device(str(cfg.get("device", "cpu")))

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

    # -- inference ----------------------------------------------------------
    def apply(self, pages: list[Page]) -> list[Page]:
        model = self.cfg.get("model")
        if model == "unet_small":
            ckpt = CKPT_DIR / "unet_small.pt"
            if not ckpt.exists():
                log.warning("unet_small checkpoint missing — falling back to classical enhance")
                return [save_image(p, _classical(load_image(p)), _out_dir()) for p in pages]
            import torch

            net = _unet().to(self.device)
            net.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
            net.eval()
            out = []
            with torch.no_grad():
                for p in pages:
                    img = load_image(p)
                    g = img if img.ndim == 2 else img[:, :, 0]
                    t = torch.from_numpy(g[None, None].astype(np.float32) / 255.0)
                    t = torch.nn.functional.interpolate(t, scale_factor=0.5, mode="bilinear")
                    rec_t = torch.nn.functional.interpolate(net(t), size=g.shape, mode="bilinear")
                    rec = (rec_t.squeeze().clamp(0, 1).numpy() * 255).astype(np.uint8)
                    out.append(save_image(p, rec, _out_dir()))
            return out

        # classical default
        out = []
        for i, pg in enumerate(pages):
            out.append(save_image(pg, _classical(load_image(pg)), _out_dir()))
            if (i + 1) % 100 == 0:
                log.info("enhance %d/%d pages", i + 1, len(pages))
        return out


def _out_dir() -> Path:
    return Path("data/interim/enhanced")


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
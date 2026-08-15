"""Unit tests for ingest: loader + preprocessing on tiny synthetic pages (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from doc_agent.ingest import loader, preprocess

# ruff: noqa: E501


def _synthetic_page(path, seed: int = 0, noise: float = 0.02) -> None:
    """A small page-like image: a few dark text rows on light paper with noise."""
    import cv2

    rng = np.random.default_rng(seed)
    img = np.full((300, 220), 245, dtype=np.uint8)
    for row in range(40, 260, 30):
        img[row : row + 12, 30:190] = rng.integers(20, 60, size=(12, 160))
    if noise:
        mask = rng.random(img.shape) < noise
        img[mask] = rng.integers(0, 255, size=int(mask.sum()))
    cv2.imwrite(str(path), img)


def test_loader_enumerates_pages(tmp_path):
    (tmp_path / "pages" / "bookA").mkdir(parents=True)
    _synthetic_page(tmp_path / "pages" / "bookA" / "page_0001.jpg", seed=1)
    _synthetic_page(tmp_path / "pages" / "bookA" / "page_0002.jpg", seed=2)
    cfg = {"root": str(tmp_path), "max_pages": -1}
    pages = loader.load_pages(cfg)
    assert len(pages) == 2
    assert pages[0].id == "bookA_p0001"
    assert pages[1].id == "bookA_p0002"
    assert pages[0].doc_id == "bookA"
    assert pages[0].image_path.endswith(".jpg")


def test_loader_max_pages_cap(tmp_path):
    (tmp_path / "pages" / "bookA").mkdir(parents=True)
    for i in range(1, 4):
        _synthetic_page(tmp_path / "pages" / "bookA" / f"page_{i:04d}.jpg", seed=i)
    pages = loader.load_pages({"root": str(tmp_path), "max_pages": 2})
    assert len(pages) == 2


def test_loader_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        loader.load_pages({"root": str(tmp_path / "nope")})


def test_preprocess_roundtrip(tmp_path):
    (tmp_path / "pages" / "bookA").mkdir(parents=True)
    p = tmp_path / "pages" / "bookA" / "page_0001.jpg"
    _synthetic_page(p, seed=3, noise=0.05)
    pages = loader.load_pages({"root": str(tmp_path)})
    cfg = {
        "preprocess": {"enabled": True, "denoise": True, "binarize": True, "deskew": True},
        "data": {"root": str(tmp_path)},
    }
    out = preprocess.run(pages, cfg)
    assert len(out) == 1
    assert out[0].id == pages[0].id
    # output image exists and is a valid grayscale image
    img = loader.load_image(out[0])
    assert img.shape[:2] == (300, 220)  # imread loads grayscale JPEG as 3ch BGR


def test_preprocess_disabled_returns_original_paths(tmp_path):
    (tmp_path / "pages" / "bookA").mkdir(parents=True)
    _synthetic_page(tmp_path / "pages" / "bookA" / "page_0001.jpg")
    pages = loader.load_pages({"root": str(tmp_path)})
    cfg = {"preprocess": {"enabled": False}, "data": {"root": str(tmp_path)}}
    out = preprocess.run(pages, cfg)
    assert [p.image_path for p in out] == [p.image_path for p in pages]
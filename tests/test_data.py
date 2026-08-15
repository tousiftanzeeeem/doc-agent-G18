"""Unit tests for data validation and corpus versioning."""

from __future__ import annotations

import pytest

from doc_agent.contracts import Page
from doc_agent.data import validate, versioning


def _page(i: int, path: str) -> Page:
    return Page(id=f"d_p{i:04d}", image_path=path, doc_id="d")


def test_validate_ok(tmp_path):
    f = tmp_path / "p.jpg"
    f.write_bytes(b"x")
    pages = [_page(1, str(f)), _page(2, str(f))]
    validate.validate(pages)  # no raise


def test_validate_empty_raises():
    with pytest.raises(ValueError):
        validate.validate([])


def test_validate_duplicate_ids_raises(tmp_path):
    f = tmp_path / "p.jpg"
    f.write_bytes(b"x")
    p1 = _page(1, str(f))
    p2 = Page(id="d_p0001", image_path=str(f), doc_id="d")
    with pytest.raises(ValueError):
        validate.validate([p1, p2])


def test_validate_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        validate.validate([_page(1, "no/such/file.jpg")])


def test_snapshot_is_stable_and_records(tmp_path, monkeypatch):
    monkeypatch.setattr(versioning, "VERSION_FILE", tmp_path / "version.json")
    (tmp_path / "a.jpg").write_bytes(b"abc")
    (tmp_path / "b.jpg").write_bytes(b"def")
    v1 = versioning.snapshot(str(tmp_path))
    v2 = versioning.snapshot(str(tmp_path))
    assert v1 == v2  # deterministic
    assert (tmp_path / "version.json").exists()
    (tmp_path / "c.jpg").write_bytes(b"ghi")
    v3 = versioning.snapshot(str(tmp_path))
    assert v3 != v1  # corpus changed -> version changed


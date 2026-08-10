"""OPTIONAL — generate synthetic text-lines for tiny-label training
Activate only if your data speciality or NFR requires it (e.g. a tiny-label / data-efficient project). Off by default; CI does not require impl."""
from __future__ import annotations

def generate(cfg: dict) -> list:
    raise NotImplementedError("optional: synthetic data")


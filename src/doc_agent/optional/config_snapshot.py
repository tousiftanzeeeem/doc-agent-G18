"""OPTIONAL — snapshot exact run config for reproducibility+
Activate only if your data speciality or NFR requires it (repro+). Off by default; CI does not require impl."""
from __future__ import annotations

def snapshot(cfg: dict, out: str) -> None:
    raise NotImplementedError("optional: config snapshot")


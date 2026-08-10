"""OPTIONAL — query embedding + expansion
Activate only if your data speciality or NFR requires it (profiles 18, 27). Off by default; CI does not require impl."""
from __future__ import annotations

def expand(query: str, cfg: dict) -> list[str]:
    raise NotImplementedError("optional: query expansion")


"""OPTIONAL — automated retrain on drift/SLA breach
Activate only if your data speciality or NFR requires it (MLOps closed-loop). Off by default; CI does not require impl."""
from __future__ import annotations

def maybe_retrain(metrics: dict, cfg: dict) -> bool:
    raise NotImplementedError("optional: retrain trigger")


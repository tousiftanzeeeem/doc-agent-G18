"""OPTIONAL — differential privacy (DP-SGD) for shared models
Activate only if your data speciality or NFR requires it (e.g. a PII-sensitive corpus / private NFR). Off by default; CI does not require impl."""
from __future__ import annotations

def make_private(optimizer, cfg: dict):
    raise NotImplementedError("optional: DP-SGD")


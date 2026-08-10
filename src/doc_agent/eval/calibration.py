"""Stage 9 — confidence calibration (calibrated-confidence NFR)"""
from __future__ import annotations
from ..contracts import *  # noqa

def temperature_scale(logits, labels):
    """Fit temperature on val; return scaler. IMPLEMENT."""
    raise NotImplementedError("Calibration: temperature scaling")
def ece(confidences, correct) -> float:
    raise NotImplementedError("Calibration: ECE")


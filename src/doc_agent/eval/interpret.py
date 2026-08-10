"""Stage 9 — EXPLAINABLE — why-retrieved / where-looked"""
from __future__ import annotations
from ..contracts import *  # noqa

def explain(answer: Answer, cfg: dict) -> dict:
    """Grad-CAM on read region + retrieval-score attribution. IMPLEMENT."""
    raise NotImplementedError("Stage 9: interpretability")


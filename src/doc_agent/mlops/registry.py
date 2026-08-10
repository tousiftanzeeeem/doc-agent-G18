"""MLOps — model registry"""
from __future__ import annotations
from ..contracts import *  # noqa

def register(component: str, path: str, metrics: dict) -> str:
    """Version a checkpoint + its metrics. IMPLEMENT."""
    raise NotImplementedError("MLOps: register")


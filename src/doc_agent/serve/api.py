"""Stage 8 — FastAPI service"""
from __future__ import annotations
from ..contracts import *  # noqa

from fastapi import FastAPI
from .. import config, pipeline

app = FastAPI(title="doc-agent")
_cfg = config.load()

@app.post("/answer")
def answer(q: str) -> dict:
    """Return grounded, cited answer. IMPLEMENT (calls pipeline.answer)."""
    raise NotImplementedError("Stage 8: /answer endpoint")

@app.get("/health")
def health() -> dict:
    return {"ok": True}


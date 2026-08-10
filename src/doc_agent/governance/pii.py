"""Governance — PII detection + redaction (mandatory)"""
from __future__ import annotations
from ..contracts import *  # noqa

def detect(text: str) -> list[tuple[int,int,str]]:
    """Return (start,end,type) PII spans. IMPLEMENT."""
    raise NotImplementedError("Governance: PII detect")
def redact(text: str) -> str:
    raise NotImplementedError("Governance: PII redact")



def register(hooks) -> None:
    """Wire PII redaction into the pipeline. IMPLEMENT the handler (call redact())."""
    def _scrub(ctx: dict) -> dict:
        raise NotImplementedError("PII: redact text/answer/log in ctx")
    hooks.register(hooks.AFTER_OCR, _scrub)       # scrub extracted text before indexing
    hooks.register(hooks.BEFORE_ANSWER, _scrub)   # scrub the outgoing answer
    hooks.register(hooks.ON_LOG, _scrub)          # scrub logs

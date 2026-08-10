"""LLM — answer post-process / format / abstention"""
from __future__ import annotations
from ..contracts import *  # noqa

def format_answer(raw: str, citations: list) -> Answer:
    """Attach citations, set grounded/confidence, enforce abstention. IMPLEMENT."""
    raise NotImplementedError("LLM: format_answer")



def register(hooks) -> None:
    """Wire the grounding / abstention gate. IMPLEMENT (abstain if answer unsupported by evidence)."""
    def _ground(ctx: dict) -> dict:
        raise NotImplementedError("Grounding: abstain if unsupported; enforce citations")
    hooks.register(hooks.BEFORE_ANSWER, _ground)

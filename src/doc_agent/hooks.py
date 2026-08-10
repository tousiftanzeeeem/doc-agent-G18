"""FIXED — cross-cutting seam. Horizontal features register handlers here; the pipeline and agent
call run() at fixed seams. Do NOT add/remove seams or the hooks.run() calls that use them."""
from __future__ import annotations
from collections import defaultdict
from typing import Callable

# The only points where cross-cutting code runs.
AFTER_INGEST = "after_ingest"
AFTER_OCR = "after_ocr"
BEFORE_INDEX = "before_index"
AFTER_RETRIEVE = "after_retrieve"
ON_STEP = "on_step"
ON_TOOL_CALL = "on_tool_call"
BEFORE_ANSWER = "before_answer"
AFTER_ANSWER = "after_answer"
ON_LOG = "on_log"
SEAMS = [AFTER_INGEST, AFTER_OCR, BEFORE_INDEX, AFTER_RETRIEVE, ON_STEP,
         ON_TOOL_CALL, BEFORE_ANSWER, AFTER_ANSWER, ON_LOG]

_handlers: dict[str, list[Callable]] = defaultdict(list)

def register(seam: str, handler: Callable) -> None:
    """Attach a handler to a seam. Called by each feature's register() via wiring.py."""
    assert seam in SEAMS, f"unknown seam {seam}"
    _handlers[seam].append(handler)

def run(seam: str, ctx: dict) -> dict:
    """Run every handler registered at `seam`, threading ctx through. Called at fixed points only."""
    for h in _handlers[seam]:
        ctx = h(ctx) or ctx
    return ctx

def clear() -> None:
    _handlers.clear()

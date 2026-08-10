"""FIXED — the single auditable manifest of cross-cutting features and where they attach.
register_all() wires every horizontal feature into the hook seams. If a feature is not listed here,
it is not wired anywhere. Implement each feature's register() in its owning module."""
from __future__ import annotations
from . import hooks, logging_conf
from .governance import pii
from .agent import guardrails
from .llm import postprocess

def register_all(cfg: dict) -> None:
    hooks.clear()
    logging_conf.register(hooks)        # tracing/audit  -> ON_STEP, ON_TOOL_CALL, AFTER_ANSWER
    pii.register(hooks)                 # privacy        -> AFTER_OCR, BEFORE_ANSWER, ON_LOG
    guardrails.register(hooks, cfg)     # security       -> ON_TOOL_CALL
    postprocess.register(hooks)         # grounding      -> BEFORE_ANSWER

"""FIXED — structured logging (auditable NFR). Use get_logger(), never print()."""
from __future__ import annotations
import logging, sys

def get_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter('{"ts":"%(asctime)s","lvl":"%(levelname)s","mod":"%(name)s","msg":"%(message)s"}'))
        lg.addHandler(h); lg.setLevel(logging.INFO)
    return lg


def register(hooks) -> None:
    """Wire structured tracing at each seam (auditable trail) AND emit traces/run.jsonl.
    Each seam appends a contracts.TraceStep line to traces/run.jsonl so the A3 agentic-feature
    check can read the trajectory (path must depend on observations). IMPLEMENT."""
    def _trace(ctx: dict) -> dict:
        raise NotImplementedError("Tracing: append ctx to the audit trail")
    hooks.register(hooks.ON_STEP, _trace)
    hooks.register(hooks.ON_TOOL_CALL, _trace)
    hooks.register(hooks.AFTER_ANSWER, _trace)

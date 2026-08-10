"""Stage 6 — SECURITY — autonomy, budgets, prompt-injection defense"""
from __future__ import annotations
from ..contracts import *  # noqa

class Guardrails:
    """Enforce autonomy level, step/cost budget, and instruction/content isolation."""
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["agent"]
    def reset(self) -> None:
        self.spent = 0.0; self.steps = 0
    def check(self, action: dict) -> None:
        """Raise if over budget / disallowed autonomy / injection detected. IMPLEMENT."""
        raise NotImplementedError("Stage 6: guardrails")



def register(hooks, cfg: dict) -> None:
    """Wire guardrails into every tool call. IMPLEMENT (call Guardrails.check)."""
    g = Guardrails(cfg); g.reset()
    def _check(ctx: dict) -> dict:
        g.check(ctx["action"])                    # budgets / autonomy / injection
        return ctx
    hooks.register(hooks.ON_TOOL_CALL, _check)

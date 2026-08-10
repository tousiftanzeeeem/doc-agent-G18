"""Stage 9 — LLM-as-judge for non-verifiable inference"""
from __future__ import annotations
from ..contracts import *  # noqa

def judge(query: Query, answer: Answer) -> float:
    """Score open-ended answers (causal/summary/intent). IMPLEMENT."""
    raise NotImplementedError("Stage 9: LLM-as-judge")


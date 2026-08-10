"""HITL — human-in-the-loop review queue"""
from __future__ import annotations
from ..contracts import *  # noqa

def escalate(reason: str, context: dict) -> ToolResult:
    """Queue for human review; block action until approved. IMPLEMENT."""
    raise NotImplementedError("HITL: escalate_to_human")

def review_queue():
    """Return pending items for the reviewer UI. IMPLEMENT."""
    raise NotImplementedError("HITL: review queue")


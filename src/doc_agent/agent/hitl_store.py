"""HITL — persistent review queue (survives restarts)"""
from __future__ import annotations
from ..contracts import *  # noqa

def enqueue(item: dict) -> str:
    """Persist a pending review item; return id. IMPLEMENT (sqlite/json)."""
    raise NotImplementedError("HITL store: enqueue")
def pending() -> list[dict]:
    raise NotImplementedError("HITL store: pending")
def resolve(item_id: str, decision: str) -> None:
    raise NotImplementedError("HITL store: resolve")


"""Stage 5 — dense retrieval"""
from __future__ import annotations
from ..contracts import *  # noqa

class Retriever:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["retrieve"]
    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        """Top-k dense retrieval. Set chunk.score (relevance) on every result so decide() can judge
        whether the evidence is weak. IMPLEMENT."""
        raise NotImplementedError("Stage 5: retrieve")


# --- evidence-strength policy: read by agent.decide() for evidence-gated re-search ---
def top_score(chunks: list[Chunk]) -> float:
    """Strength of the current evidence = best chunk score (0.0 if empty)."""
    return max((c.score for c in chunks), default=0.0)

def is_weak(chunks: list[Chunk], cfg: dict) -> bool:
    """Weak evidence = best score below cfg.retrieve.weak_threshold."""
    return top_score(chunks) < cfg["retrieve"]["weak_threshold"]

def next_k(k: int, cfg: dict) -> int | None:
    """Widen the net: k + k_step, or None once it would exceed k_max (signal to ABSTAIN)."""
    nk = k + cfg["retrieve"]["k_step"]
    return nk if nk <= cfg["retrieve"]["k_max"] else None


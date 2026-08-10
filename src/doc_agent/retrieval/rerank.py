"""Stage 5 — reranking"""
from __future__ import annotations
from ..contracts import *  # noqa

def rerank(query: str, candidates: list[Chunk], cfg: dict) -> list[Chunk]:
    """Cross-encoder rerank if cfg['retrieve']['rerank']. IMPLEMENT."""
    raise NotImplementedError("Stage 5: rerank")


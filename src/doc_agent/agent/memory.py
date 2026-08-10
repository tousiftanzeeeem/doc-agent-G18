"""Stage 6 — working/episodic memory"""
from __future__ import annotations
from ..contracts import *  # noqa

class Memory:
    def __init__(self) -> None:
        self.items: list = []
    def add(self, item) -> None:
        self.items.append(item)
    def recall(self, query: str) -> list:
        raise NotImplementedError("Stage 6: memory recall")


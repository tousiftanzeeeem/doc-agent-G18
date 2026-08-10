"""Stage 7 — policy network"""
from __future__ import annotations
from ..contracts import *  # noqa

class Policy:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["rl"]
    def act(self, state) -> dict:
        raise NotImplementedError("Stage 7: policy.act")


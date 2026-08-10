"""Stage 6 - FIXED loop - perceive -> decide -> act -> observe, with cross-cutting seams.
Implement decide() and synthesize() only. Security, grounding, PII, and tracing run via hooks at the
marked seams - do NOT inline them here."""
from __future__ import annotations
from ..contracts import *  # noqa
from .. import hooks
from .memory import Memory

class Agent:
    """FIXED loop. Implement decide() (the policy) and synthesize() only."""
    def __init__(self, cfg: dict, retriever) -> None:
        self.cfg = cfg["agent"]; self.retriever = retriever; self.mem = Memory()

    def run(self, query_text: str) -> Answer:
        state = {"query": query_text, "obs": []}
        for _ in range(self.cfg["max_steps"]):
            hooks.run(hooks.ON_STEP, {"state": state})
            action = self.decide(state)                      # IMPLEMENT (policy)
            if action["tool"] == "stop":
                break
            hooks.run(hooks.ON_TOOL_CALL, {"action": action})   # guardrails/injection/trace
            result = self.act(action)                        # runs the tool via REGISTRY
            state["obs"].append(result); self.mem.add(result)
        hooks.run(hooks.BEFORE_ANSWER, {"state": state})     # grounding gate / PII redact
        ans = self.synthesize(state)                         # IMPLEMENT (grounded answer)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})       # trace / metrics
        return ans

    def decide(self, state: dict) -> dict:
        """Evidence-gated re-search — the MANDATORY agentic behaviour (A3 gate, fail-closed).
        Read the last observation (top_score, k) and branch on the NUMBER, using retrieval.retriever:
          1. retrieve at k = cfg.retrieve.k
          2. if is_weak(chunks, cfg):  k2 = next_k(k, cfg)
               - k2 is not None -> retrieve AGAIN at the wider k2 (widen the net), then re-check
               - k2 is None (hit k_max) and still weak -> ABSTAIN ("insufficient evidence")
          3. else -> synthesize a grounded, cited answer
        Emit obs {"top_score": ..., "k": ...} on each step. A fixed retrieve->answer path is NOT agentic
        and caps the grade. Rule-based (baseline) or RL policy (Stage 7)."""
        raise NotImplementedError("Stage 6: agent policy")

    def act(self, action: dict) -> ToolResult:
        raise NotImplementedError("Stage 6: dispatch tool from tools.REGISTRY")

    def synthesize(self, state: dict) -> Answer:
        """Grounded, cited answer; abstain if unsupported (no-hallucination)."""
        raise NotImplementedError("Stage 6: synthesize grounded answer")

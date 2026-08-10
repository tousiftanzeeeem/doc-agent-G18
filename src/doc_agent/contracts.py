"""FIXED data contracts. Do not change fields or types."""
from __future__ import annotations
from pydantic import BaseModel

class Page(BaseModel):
    id: str
    image_path: str
    doc_id: str

class Region(BaseModel):
    page_id: str
    bbox: tuple[int, int, int, int]
    kind: str  # text | table | figure | heading

class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    page_ids: list[str]
    score: float = 0.0        # relevance score set by retrieval; decide() reads the top score to judge weak evidence

class Query(BaseModel):
    text: str
    verifiable: bool          # True = checkable by exact match (anchors objective grading + RLVR)
    judged: bool              # True = scored by LLM-judge / human (non-verifiable inference)

class Citation(BaseModel):
    chunk_id: str
    span: tuple[int, int]

class Answer(BaseModel):
    text: str
    citations: list[Citation]
    grounded: bool
    confidence: float

class ToolResult(BaseModel):
    ok: bool
    payload: dict


class TraceStep(BaseModel):
    """One agent step, emitted to traces/run.jsonl so the A3 agentic-feature check can read the path."""
    step: int
    tool: str                 # tool name, or "decide" / "answer"
    args: dict                # e.g. {"query": "..."} — for retrieve, the query actually used
    obs: dict                 # what decide() saw, e.g. {"top_score": 0.31, "n": 10}

# MANDATORY agentic behaviour (graded in A3): evidence-gated re-search — decide() must re-retrieve with a
# reformulated query when the top evidence is weak. That runtime branch is what makes the system an agent.
# (multi-hop -> recall NFR, verify-and-correct -> precision NFR, tool-routing -> bonus; see the codebase guide.)


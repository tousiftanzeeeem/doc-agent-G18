"""LLM — FIXED prompt template registry (all prompts live here)"""
from __future__ import annotations
from ..contracts import *  # noqa

# Fill the template bodies; do NOT scatter prompt strings elsewhere.
DECIDE = "IMPLEMENT: tool-selection prompt"
SYNTHESIZE = "IMPLEMENT: grounded-answer prompt (must force citations + abstention)"
JUDGE = "IMPLEMENT: LLM-as-judge prompt for open-ended inference"


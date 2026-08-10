"""Stage 8 — affordable adaptation — LoRA / quantization"""
from __future__ import annotations
from ..contracts import *  # noqa

def apply_lora(model, cfg: dict):
    """Wrap a component with LoRA per cfg. IMPLEMENT."""
    raise NotImplementedError("Adapt: LoRA")
def quantize(model, cfg: dict):
    """Post-training quantization per cfg. IMPLEMENT."""
    raise NotImplementedError("Adapt: quantize")


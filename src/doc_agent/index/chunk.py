"""Stage 4 — chunk text"""
from __future__ import annotations
from ..contracts import *  # noqa
from transformers import AutoTokenizer

# Model name from your config.yaml: embed.model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def test_tokenizer():
    print(f"Loading tokenizer for model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    sample_text = (
        "Document processing and retrieval-augmented generation (RAG) pipelines "
        "rely heavily on accurate token-based chunking."
    )

    # 1. Test encoding (matching chunk.py behavior)
    token_ids = tokenizer.encode(sample_text, add_special_tokens=False)
    print(f"\n[PASS] Encoding successful!")
    print(f"       Total tokens: {len(token_ids)}")
    print(f"       First 10 token IDs: {token_ids[:10]}")

    # 2. Test decoding (matching chunk.py behavior)
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"\n[PASS] Decoding successful!")
    print(f"       Decoded text: '{decoded_text}'")

    # 3. Quick verification
    assert len(token_ids) > 0, "Token IDs list should not be empty"
    print("\n✅ Tokenizer is working correctly!")

test_tokenizer()


# def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
#     """Re-chunk to cfg['index'] size/overlap. IMPLEMENT."""
#     raise NotImplementedError("Stage 4: chunk")


"""Stage 4 — chunk text"""
# if we only wanna run this file, we can do so with  `PYTHONPATH=src python3 -m doc_agent.index.chunk` and it will run this python file as a module. This is useful for testing the chunking functionality in isolation.
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

# test_tokenizer()


def _get_tokenizer(cfg: dict):
    model_name = cfg["embed"]["model"]
    return AutoTokenizer.from_pretrained(model_name)

def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Re-chunk to cfg['index'] size/overlap.
 
    Each input Chunk (one OCR'd layout region) is windowed independently:
    region boundaries are never crossed. Uses the embedding model's own
    tokenizer so 'chunk_tokens' means the same thing here as it will to embed.py.
    """
    tokenizer = _get_tokenizer(cfg)
    chunk_tokens = cfg["index"]["chunk_tokens"]
    overlap = cfg["index"]["overlap"]
    stride = chunk_tokens - overlap
    assert stride > 0, "overlap must be smaller than chunk_tokens"
 
    out: list[Chunk] = []
    for region_chunk in chunks:
        token_ids = tokenizer.encode(region_chunk.text, add_special_tokens=False)
 
        # Region already fits in one window -> keep as-is (still re-wrapped for a fresh id).
        if len(token_ids) <= chunk_tokens:
            out.append(Chunk(
                id=f"{region_chunk.id}-0",
                doc_id=region_chunk.doc_id,
                text=region_chunk.text,
                page_ids=region_chunk.page_ids,
            ))
            continue
 
        start = 0
        window_idx = 0
        while start < len(token_ids):
            end = min(start + chunk_tokens, len(token_ids))
            window_ids = token_ids[start:end]
            window_text = tokenizer.decode(window_ids, skip_special_tokens=True)
 
            out.append(Chunk(
                id=f"{region_chunk.id}-{window_idx}",
                doc_id=region_chunk.doc_id,
                text=window_text,
                page_ids=region_chunk.page_ids,
            ))
 
            if end == len(token_ids):
                break
            start += stride
            window_idx += 1
 
    return out


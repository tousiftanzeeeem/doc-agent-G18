"""Stage 4 — embed chunks"""
# run only this file as a module as "PYTHONPATH=src python3 -m doc_agent.index.embed"
from __future__ import annotations
from ..contracts import *  # noqa
from sentence_transformers import SentenceTransformer

def _get_model(cfg: dict) -> "SentenceTransformer":
    model_name = cfg["embed"]["model"]
    return SentenceTransformer(model_name)

def encode(chunks: list[Chunk], cfg: dict) -> list[list[float]]:
    """Embed with cfg['embed']['model'].
 
    Returns a list of vectors, one per chunk, in the same order as `chunks`
    (i.e. vectors[i] is the embedding of chunks[i]). Each vector has
    cfg['embed']['dim'] numbers in it.
    """
    model = _get_model(cfg)
    texts = [c.text for c in chunks]
 
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit-length vectors -> cosine sim == dot product later
        show_progress_bar=False,
    )
 
    expected_dim = cfg["embed"]["dim"]
    assert embeddings.shape[1] == expected_dim, (
        f"Model produced dim {embeddings.shape[1]}, but cfg says {expected_dim}"
    )
 
    return embeddings.tolist()

def _test_encode():
    from ..config import load
    """Quick sanity check for the embedding model."""
    chunks = [
        Chunk(id="test-0", doc_id="test-doc", text="Hello world!", page_ids=["test-page"], score=0.0),
        Chunk(id="test-1", doc_id="test-doc", text="The quick brown fox jumps over the lazy dog.", page_ids=["test-page"], score=0.0),
    ]
    cfg = load()
    embeddings = encode(chunks, cfg)
    print("embeddings are : ")
    for i, emb in enumerate(embeddings):
        print(f"Chunk {i} embedding (length {len(emb)}): {emb[:5]}...")  # print first 5 values

if __name__ == "__main__":
    _test_encode()
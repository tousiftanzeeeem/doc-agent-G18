"""Stage 4 — vector store"""
from __future__ import annotations
from ..contracts import *  # noqa
import json
import os
import numpy as np
import faiss


def _paths(cfg: dict) -> tuple[str, str]:
    """Where the FAISS index and the chunk-metadata sidecar file live on disk."""
    index_dir = cfg["index"].get("dir", "data/index")
    os.makedirs(index_dir, exist_ok=True)
    return (
        os.path.join(index_dir, "vectors.faiss"),
        os.path.join(index_dir, "chunks.jsonl"),
    )


def build(chunks: list[Chunk], vectors: list[list[float]], cfg: dict) -> None:
    """Persist a vector index (cfg['index']['type']).

    Stores two things, kept in sync by row position:
      1. vectors.faiss  - the FAISS index itself (numbers only, for fast search)
      2. chunks.jsonl   - the chunk metadata (id/doc_id/text/page_ids), one row
                           per vector, in the same order, so a FAISS search hit
                           at row i can be mapped back to chunks[i].
    """
    assert len(chunks) == len(vectors), "chunks and vectors must be parallel lists"

    index_path, meta_path = _paths(cfg)
    dim = cfg["embed"]["dim"]

    mat = np.array(vectors, dtype="float32")
    assert mat.shape == (len(chunks), dim), f"expected ({len(chunks)}, {dim}), got {mat.shape}"

    index_type = cfg["index"]["type"]
    if index_type == "faiss:hnsw":
        # metric_type=INNER_PRODUCT (not the default L2) because embed.py
        # normalizes every vector to unit length -> inner product == cosine similarity.
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    elif index_type == "faiss:flat":
        index = faiss.IndexFlatIP(dim)  # exact brute-force search, inner product
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    index.add(mat)
    faiss.write_index(index, index_path)

    with open(meta_path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c.model_dump()) + "\n")


def load(cfg: dict):
    """Load the FAISS index + chunk metadata back into memory.

    Returns (index, chunks) where chunks[i] is the metadata for the vector
    at row i of the index - the same alignment build() wrote.
    """
    index_path, meta_path = _paths(cfg)

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No index found at {index_path} / {meta_path}. Run build() first."
        )

    index = faiss.read_index(index_path)

    chunks: list[Chunk] = []
    with open(meta_path) as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))

    return index, chunks

def _test_store() :
    """Quick sanity check for the store."""
    from ..config import load as load1
    from .chunk import split
    from .embed import encode

    cfg = load1()
    chunks = [
        Chunk(id="test-0", doc_id="test-doc", text="Hello world!", page_ids=["test-page"], score=0.0),
        Chunk(id="test-1", doc_id="test-doc", text="The quick brown fox jumps over the lazy dog.", page_ids=["test-page"], score=0.0),
    ]
    vectors = encode(chunks, cfg)
    build(chunks, vectors, cfg)

    index, loaded_chunks = load(cfg)
    assert len(loaded_chunks) == len(chunks), "Loaded chunks count mismatch"
    assert index.ntotal == len(vectors), "Loaded index count mismatch"
    print("Store test passed.")

if __name__ == "__main__":
    _test_store()
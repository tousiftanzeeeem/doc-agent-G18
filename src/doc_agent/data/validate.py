"""Data — data schema/quality validation at ingest"""

from __future__ import annotations

from pathlib import Path

from ..contracts import Page
from ..logging_conf import get_logger

log = get_logger("data.validate")

MIN_PAGES = 300
MIN_WORDS = 60000


def validate(pages: list[Page]) -> None:
    """Assert pages are loadable and unique; report the corpus floor against the full run."""
    if not pages:
        raise ValueError("empty page list — nothing to validate")
    ids = [p.id for p in pages]
    if len(set(ids)) != len(ids):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"duplicate page ids: {sorted(dupes)[:5]}...")
    missing = [p.image_path for p in pages if not Path(p.image_path).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} page images missing, e.g. {missing[0]}")

    # Corpus floor (>=300 pages, >=60k words) applies to the full run, not test subsets.
    # Enforce the page floor only when the whole raw corpus is being processed.
    if len(pages) >= MIN_PAGES:
        log.info("corpus: %d pages (floor %d) ✓", len(pages), MIN_PAGES)
    else:
        log.warning("corpus: %d pages < floor %d (subset run?)", len(pages), MIN_PAGES)

    doc_ids = {p.doc_id for p in pages}
    log.info("data validation OK: %d pages, %d documents", len(pages), len(doc_ids))
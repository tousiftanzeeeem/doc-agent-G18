"""Data — corpus versioning (which corpus version -> which result)"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..logging_conf import get_logger

log = get_logger("data.versioning")

VERSION_FILE = Path("data/version.json")


def snapshot(corpus_dir: str) -> str:
    """Hash the corpus file names + sizes -> a short version id, recorded to data/version.json."""
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"corpus dir {root} not found")

    entries: list[tuple[str, int]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != VERSION_FILE.name:
            entries.append((str(p.relative_to(root)), p.stat().st_size))

    h = hashlib.sha256()
    for name, size in entries:
        h.update(f"{name}:{size}\n".encode())
    version = h.hexdigest()[:12]

    VERSION_FILE.write_text(
        json.dumps(
            {
                "corpus_dir": str(root),
                "version": version,
                "files": len(entries),
                "bytes": sum(s for _, s in entries),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("corpus version %s (%d files)", version, len(entries))
    return version
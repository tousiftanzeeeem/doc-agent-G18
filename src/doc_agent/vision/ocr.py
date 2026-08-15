"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation).

Reader wraps RapidOCR (the PP-OCRv4 detection + recognition models exported to ONNX and served by
onnxruntime) — a reproduced, published pretrained recognizer. The bulk path runs page-level OCR
once per page (detection boxes + recognition) and distributes lines into the regions found by
Stage 2, so a whole page costs a single model pass. Page-level results are cached under
data/interim/ocr_cache/ so rebuilds of the index do not re-run the recognizer. When
cfg['ocr']['workers'] > 1 the pages are OCR'd by a process pool (each worker owns its recognizer)
and results land in the same cache.

Backend history (see configs/design_choices.md): PaddleOCR PP-OCRv5 was evaluated first but the
installed paddlepaddle 3.3.1 build crashes on CPU inference (PIR/oneDNN
`ConvertPirAttribute2RuntimeAttribute` NotImplementedError); EasyOCR (CRAFT+CRNN) works but is ~4x
slower and less accurate on this 1890 letterpress than RapidOCR's PP-OCRv4 ONNX models.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import Chunk, Page, Region
from ..logging_conf import get_logger

log = get_logger("vision.ocr")

_ENGINES: dict[tuple[str, ...], Any] = {}
CACHE_DIR = Path("data/interim/ocr_cache")
_THREADS = 0  # >0 when the OCR process pool caps per-worker onnxruntime threads

_CUDA_DLL_DIRS: tuple[str, ...] | None = None


def _cuda_dll_dirs() -> tuple[str, ...]:
    """Locate the pip-bundled nvidia CUDA/cuDNN DLL dirs (onnxruntime-gpu[cuda,cudnn])."""
    global _CUDA_DLL_DIRS
    if _CUDA_DLL_DIRS is None:
        try:
            import onnxruntime as ort

            sp = os.path.dirname(os.path.dirname(ort.__file__))  # site-packages
            nv = os.path.join(sp, "nvidia")
            dirs: list[str] = []
            for base in (
                os.path.join(nv, "cu13", "bin", "x86_64"),
                os.path.join(nv, "cudnn", "bin"),
                os.path.join(nv, "cu13", "bin"),
            ):
                if os.path.isdir(base):
                    dirs.append(base)
            _CUDA_DLL_DIRS = tuple(dict.fromkeys(dirs))
        except Exception:
            _CUDA_DLL_DIRS = ()
    return _CUDA_DLL_DIRS


@contextmanager
def _cuda_dll_path() -> Iterator[None]:
    """Temporarily prepend the bundled nvidia DLL dirs to PATH while onnxruntime
    loads its CUDA provider and runs inference (it resolves cublas/cudnn via PATH
    at session creation AND cuDNN 9 loads its engine sub-libraries lazily at the
    first Conv call).

    Scoped so torch's own bundled cuDNN (a CUDA-12 build) is never shadowed by the
    CUDA-13 nvidia packages afterwards — the two cuDNN versions cannot coexist in
    one process, so the parent pipeline keeps OCR on CPU and lets GPU OCR run in
    the worker processes (which never import torch).
    """
    old = os.environ.get("PATH", "")
    path = old
    for d in _cuda_dll_dirs():
        if d not in path:
            path = d + os.pathsep + path
    os.environ["PATH"] = path
    try:
        yield
    finally:
        os.environ["PATH"] = old


def _ensure_cuda_dlls_on_path() -> None:
    """Persistently prepend the nvidia DLL dirs to PATH (idempotent).

    Used only inside OCR worker processes, which never import torch, so the
    CUDA-13 cuDNN/cuBLAS can stay on PATH for the worker's whole lifetime.
    """
    path = os.environ.get("PATH", "")
    for d in _cuda_dll_dirs():
        if d not in path:
            path = d + os.pathsep + path
    os.environ["PATH"] = path

# Lazy import to avoid circular — used only inside transcribe()
_table_recovery = None

def _get_table_recovery() -> Any:
    global _table_recovery
    if _table_recovery is None:
        from ..optional import table_recovery as _tr

        _table_recovery = _tr
    return _table_recovery


def _resolve_device(requested: str) -> str:
    """'cuda' if requested and a CUDA-capable onnxruntime is installed, else 'cpu'.

    RapidOCR itself re-checks and falls back to the CPU EP when the CUDA session
    cannot be created, so this only short-circuits the obvious no-CUDA case early.
    """
    if requested != "cuda":
        return "cpu"
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "cuda"
    except Exception:
        pass
    log.warning(
        "device: cuda requested but onnxruntime has no CUDA provider — OCR falls back to cpu"
    )
    return "cpu"


class Reader:
    """Model set by cfg['ocr']. Baseline: pretrained RapidOCR (PP-OCRv4 ONNX)."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]
        self.device = _resolve_device(str(cfg.get("device", "cpu")))
        self._engine = None

    def _get_engine(self) -> Any:
        key: tuple[Any, ...] = (
            str(self.cfg.get("model", "rapidocr:PP-OCRv4-onnx")),
            _THREADS,
            self.device,
        )
        if key not in _ENGINES:
            from rapidocr_onnxruntime import RapidOCR

            kw: dict[str, Any] = {}
            if _THREADS > 0:
                kw["intra_op_num_threads"] = _THREADS
            if self.device == "cuda":
                # RapidOCR only honours per-module flags (det_/cls_/rec_use_cuda);
                # a bare `use_cuda` lands in Global and never reaches the sessions.
                kw["det_use_cuda"] = True
                kw["cls_use_cuda"] = True
                kw["rec_use_cuda"] = True
                with _cuda_dll_path():  # onnxruntime resolves cublas/cudnn via PATH here
                    _ENGINES[key] = RapidOCR(**kw)
            else:
                _ENGINES[key] = RapidOCR(**kw)
            log.info(
                "created RapidOCR engine %s (device=%s, threads=%s)",
                key[0],
                self.device,
                _THREADS or "default",
            )
        return _ENGINES[key]

    def _infer(self, img: np.ndarray) -> Any:
        """Run the engine, keeping the nvidia DLL dirs on PATH for the call.

        cuDNN 9 loads its engine sub-libraries lazily at the first Conv, so the
        PATH patch must cover inference, not just session creation. In worker
        processes the dirs are already on PATH persistently (no-op here).
        """
        if self.device != "cuda":
            return self._get_engine()(img)
        with _cuda_dll_path():
            return self._get_engine()(img)

    # -- single-region (used by the read_page / enhance_page tools in A3) ------
    def transcribe_region(self, region: Region) -> str:
        img = _page_image(region.page_id)
        x1, y1, x2, y2 = region.bbox
        crop = img[y1:y2, x1:x2]
        res, _ = self._infer(crop)
        return _join_lines(_line_boxes(res or []))

    # -- page-level ------------------------------------------------------------
    def page_lines(self, page: Page) -> list[dict]:
        """(box, text, conf) lines for one page, from cache when possible."""
        return self.page_lines_cached(page)

    def page_lines_cached(self, page: Page) -> list[dict]:
        """Page OCR with a per-page cache (rebuilds of the index do not re-run OCR)."""
        cache = CACHE_DIR / f"{page.id}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        img = _page_image(page.id)  # current (post preprocess/enhance) image for this page
        res, _ = self._infer(img)
        lines = _line_boxes(res or [])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")
        return lines


def _limit_threads() -> None:
    """Avoid CPU oversubscription when the OCR process pool runs many workers."""
    global _THREADS
    import os

    import cv2
    import onnxruntime as ort

    workers = max(1, int(os.environ.get("OCR_WORKERS", "4")))
    _THREADS = max(1, 16 // workers)
    ort.set_default_logger_severity(3)
    cv2.setNumThreads(_THREADS)


def _line_boxes(res: Any) -> list[dict]:
    """Normalise an engine result into [{'box': [x1,y1,x2,y2], 'text': str, 'conf': float}].
    Handles both 3-tuple (box, text, conf) and 2-tuple (box, text) outputs; the same
    shape is returned by RapidOCR and EasyOCR."""
    lines: list[dict] = []
    for item in res:
        if len(item) == 3:
            poly, text, conf = item
        else:
            poly, text = item
            conf = 1.0
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        lines.append({"box": [x1, y1, x2, y2], "text": str(text), "conf": float(conf)})
    return lines


def _ocr_page_worker(args: tuple) -> tuple[str, int]:
    """Top-level pool worker: OCR one page and write its cache entry (importable on Windows)."""
    page_id, model, device = args
    _limit_threads()
    if device == "cuda":
        # workers never import torch, so the CUDA-13 nvidia DLLs can stay on PATH
        _ensure_cuda_dlls_on_path()
    reader = Reader({"ocr": {"model": model, "lang": "en"}, "device": device})
    page = Page(id=page_id, image_path="", doc_id=page_id.split("_p")[0])
    lines = reader.page_lines_cached(page)
    return page_id, len(lines)


def _ocr_pages_parallel(page_ids: list[str], cfg: dict) -> None:
    """Fill the OCR cache for the given pages using a process pool."""
    workers = int(cfg["ocr"].get("workers", 1))
    if workers <= 1 or len(page_ids) < 2:
        return
    import os

    os.environ["OCR_WORKERS"] = str(workers)  # workers read this to cap torch/cv2 threads
    args = [
        (pid, cfg["ocr"].get("model", "rapidocr:PP-OCRv4-onnx"), str(cfg.get("device", "cpu")))
        for pid in page_ids
    ]
    with Pool(workers) as pool:
        for pid, n in pool.imap_unordered(_ocr_page_worker, args, chunksize=4):
            log.info("ocr worker done %s (%d lines)", pid, n)
    log.info("parallel OCR complete: %d pages", len(page_ids))


def _join_lines(lines: list[dict]) -> str:
    ordered = sorted(lines, key=lambda ln: (ln["box"][1], ln["box"][0]))
    return "\n".join(ln["text"] for ln in ordered if ln["text"].strip())


def _page_image(page_id: str) -> np.ndarray:
    """Resolve the current (post preprocess/enhance) image for a page id, pipeline-order aware."""
    import cv2

    for base in ("data/interim/enhanced", "data/interim/processed", "data/raw"):
        p = Path(base) / f"{page_id}.jpg"
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                return img
    raise FileNotFoundError(f"no image found for page {page_id} (run ingest first)")


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Regions -> text chunks: one chunk per region, one page-level OCR pass per page.

    For table regions, runs per-cell OCR via table_recovery to produce table_cells.
    """
    # The parent process also runs torch (embed) later, and ORT's CUDA-13 cuDNN
    # cannot coexist with torch's CUDA-12 cuDNN in one process — so GPU OCR is
    # handled by the worker pool above/below and the parent stays on CPU.
    reader = Reader({**cfg, "device": "cpu"})
    by_page: dict[str, list[Region]] = {}
    for region in regions:
        by_page.setdefault(region.page_id, []).append(region)

    # warm the OCR cache in parallel first (each worker owns its recognizer)
    missing = [pid for pid in by_page if not (CACHE_DIR / f"{pid}.json").exists()]
    if missing:
        _ocr_pages_parallel(missing, cfg)

    chunks: list[Chunk] = []
    for page_id, page_regions in by_page.items():
        page = Page(id=page_id, image_path="", doc_id=page_id.split("_p")[0])
        lines = reader.page_lines(page)
        region_text: dict[int, list[str]] = {i: [] for i in range(len(page_regions))}
        for ln in lines:
            bx1, by1, bx2, by2 = ln["box"]
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            for ri, r in enumerate(page_regions):
                x1, y1, x2, y2 = r.bbox
                if x1 - 4 <= cx <= x2 + 4 and y1 - 4 <= cy <= y2 + 4:
                    region_text[ri].append(ln["text"])
                    break
        for ri, region in enumerate(page_regions):
            text = "\n".join(t for t in region_text[ri] if t.strip())

            # --- per-cell OCR for table regions (A1 commitment) ---
            if region.kind == "table":
                try:
                    tr = _get_table_recovery()
                    img = _page_image(page_id)
                    import cv2

                    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cell_grid = tr.recover_cells(gray, region.bbox)
                    if cell_grid and len(cell_grid[0]) >= 2:
                        table_cells = []
                        for row in cell_grid:
                            row_texts: list[str] = []
                            for cx1, cy1, cx2, cy2 in row:
                                # Retrieve text overlapping this cell without re-running OCR
                                cell_text = []
                                for ln in lines:
                                    bx1, by1, bx2, by2 = ln["box"]
                                    mid_x, mid_y = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                                    # If word center is inside the cell bbox
                                    if cx1 - 2 <= mid_x <= cx2 + 2 and cy1 - 2 <= mid_y <= cy2 + 2:
                                        cell_text.append(ln["text"])
                                row_texts.append(" ".join(cell_text))
                            table_cells.append(row_texts)
                        # Rebuild text from cell grid for better structure
                        text = "\n".join(
                            " | ".join(cell for cell in row)
                            for row in table_cells
                        )
                except Exception as exc:
                    log.warning("table cell recovery failed for %s:r%d: %s — using page-level OCR",
                                page_id, ri, exc)

            chunks.append(
                Chunk(
                    id=f"{page_id}:r{ri}",
                    doc_id=page_id.split("_p")[0],
                    text=text,
                    page_ids=[page_id],
                )
            )
    log.info("ocr: %d chunks from %d regions", len(chunks), len(regions))
    return chunks

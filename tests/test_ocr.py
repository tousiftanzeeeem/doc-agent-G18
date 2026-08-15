"""Unit tests for OCR: line normalisation and region->chunk mapping (engine mocked)."""

from __future__ import annotations

from doc_agent.contracts import Page, Region
from doc_agent.vision import ocr


def _fake_lines():
    # rapidocr/ocr result shape: [[poly(4 pts), text, conf], ...]
    return [
        ([[10, 20], [100, 20], [100, 30], [10, 30]], "hello", 0.99),
        ([[10, 40], [90, 40], [90, 50], [10, 50]], "world", 0.95),
    ]


def test_line_boxes_normalisation():
    lines = ocr._line_boxes(_fake_lines())
    assert lines[0]["text"] == "hello"
    assert lines[0]["box"] == [10.0, 20.0, 100.0, 30.0]
    assert lines[0]["conf"] == 0.99
    assert len(lines) == 2


def test_line_boxes_two_tuple_paragraph():
    res = [
        (
            [[0, 0], [50, 0], [50, 20], [0, 20]],
            "para text",
        )
    ]
    lines = ocr._line_boxes(res)
    assert lines[0]["text"] == "para text"
    assert lines[0]["conf"] == 1.0


def test_join_lines_reading_order():
    lines = [
        {"box": [0, 100, 50, 110], "text": "second", "conf": 0.9},
        {"box": [0, 10, 50, 20], "text": "first", "conf": 0.9},
    ]
    assert ocr._join_lines(lines) == "first\nsecond"


def test_transcribe_maps_lines_to_regions(monkeypatch):
    page_id = "bookA_p0001"
    regions = [
        Region(page_id=page_id, bbox=(10, 10, 200, 60), kind="text"),
        Region(page_id=page_id, bbox=(10, 80, 200, 120), kind="heading"),
    ]

    def fake_page_lines(self, page: Page) -> list[dict]:
        assert page.id == page_id
        return [
            {"box": [20, 20, 180, 30], "text": "alpha", "conf": 0.99},
            {"box": [20, 35, 180, 45], "text": "beta", "conf": 0.98},
            {"box": [20, 90, 180, 100], "text": "gamma", "conf": 0.97},
        ]

    monkeypatch.setattr(ocr.Reader, "page_lines", fake_page_lines)
    cfg = {"ocr": {"model": "rapidocr:PP-OCRv4-onnx", "lang": "en"}}
    chunks = ocr.transcribe(regions, cfg)
    assert len(chunks) == 2
    assert chunks[0].page_ids == [page_id]
    assert "alpha" in chunks[0].text and "beta" in chunks[0].text
    assert chunks[1].text == "gamma"


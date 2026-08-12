"""Unit tests for eval metrics (OCR F1, recall@k, groundedness, ECE, fairness gap)."""

from __future__ import annotations

import pytest

from doc_agent.contracts import Answer, Citation
from doc_agent.eval import metrics


def test_ocr_f1_perfect_and_partial():
    assert metrics.ocr_f1("hello world", "hello world") == 1.0
    assert metrics.ocr_f1("", "") == 1.0
    assert metrics.ocr_f1("helo", "hello") < 1.0
    assert metrics.ocr_f1("xyzzy", "hello") == 0.0  # no shared characters


def test_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    assert metrics.recall_at_k(retrieved, ["a", "c"], 2) == 0.5
    assert metrics.recall_at_k(retrieved, ["a"], 1) == 1.0
    assert metrics.recall_at_k(retrieved, ["x"], 4) == 0.0


def test_groundedness_and_citation_accuracy():
    grounded = Answer(
        text="answer",
        citations=[Citation(chunk_id="c1", span=(0, 6))],
        grounded=True,
        confidence=0.9,
    )
    hallucinated = Answer(text="answer", citations=[], grounded=False, confidence=0.0)
    assert metrics.groundedness(grounded) == 1.0
    assert metrics.groundedness(hallucinated) == 0.0
    assert metrics.citation_accuracy(grounded) == 1.0
    assert metrics.citation_accuracy(hallucinated) == 0.0


def test_ece_calibrated_vs_miscalibrated():
    assert metrics.ece([1.0, 1.0, 1.0], [True, True, True]) == 0.0
    assert metrics.ece([0.9, 0.9, 0.9], [False, False, False]) > 0.8


def test_subgroup_gap():
    assert metrics.subgroup_gap({"a": [0.9, 0.8], "b": [0.5, 0.5]}) == pytest.approx(0.35)

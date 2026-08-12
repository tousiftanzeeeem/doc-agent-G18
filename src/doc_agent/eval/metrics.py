"""Stage 9 — metrics"""

from __future__ import annotations

import numpy as np

from ..contracts import Answer


def ocr_f1(pred: str, gold: str) -> float:
    """Character-level precision/recall/F1 between a transcription and its ground truth."""
    pred_chars = [c for c in pred if not c.isspace()]
    gold_chars = [c for c in gold if not c.isspace()]
    if not gold_chars:
        return 1.0 if not pred_chars else 0.0
    common = sum(min(pred_chars.count(c), gold_chars.count(c)) for c in set(gold_chars))
    precision = common / max(len(pred_chars), 1)
    recall = common / len(gold_chars)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved: list, gold: list, k: int) -> float:
    """Fraction of gold items present in the top-k retrieved set."""
    top = retrieved[:k]
    if not gold:
        return 0.0
    return sum(1 for g in gold if g in top) / len(gold)


def groundedness(answer: Answer) -> float:
    """1.0 if the answer is grounded (cites evidence), else 0.0 — no-hallucination gate."""
    return 1.0 if answer.grounded and answer.citations else 0.0


def citation_accuracy(answer: Answer) -> float:
    """Fraction of citations whose span is non-empty and within the answer text."""
    if not answer.citations:
        return 0.0
    ok = 0
    for c in answer.citations:
        s, e = c.span
        ok += 1 if 0 <= s < e <= max(len(answer.text), 1) else 0
    return ok / len(answer.citations)


def ece(confidences: list[float], correct: list[bool]) -> float:
    """Expected calibration error over 10 bins."""
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.searchsorted(bins, conf, side="right") - 1, 0, 9)
    ece_val = 0.0
    for b in range(10):
        mask = idx == b
        if not mask.any():
            continue
        ece_val += (mask.sum() / conf.size) * abs(
            float(conf[mask].mean()) - float(corr[mask].mean())
        )  # noqa: E501
    return float(ece_val)


def subgroup_gap(scores_by_group: dict) -> float:
    """Fairness: max - min mean score across subgroups."""
    if not scores_by_group:
        return 0.0
    means = [float(np.mean(v)) for v in scores_by_group.values() if len(v) > 0]
    if not means:
        return 0.0
    return float(max(means) - min(means))

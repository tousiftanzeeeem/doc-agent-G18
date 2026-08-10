"""Cross-cutting features must work END TO END, not just exist in one file.
Un-skip and implement alongside the feature. CI runs these."""
import pytest

@pytest.mark.skip(reason="implement with grounding")
def test_grounding_unsupported_query_abstains():
    """An answer with no supporting evidence must abstain, not fabricate."""
    assert True

@pytest.mark.skip(reason="implement with security")
def test_injection_in_document_does_not_hijack():
    """A document containing 'ignore your instructions' must not change agent behaviour."""
    assert True

@pytest.mark.skip(reason="implement with PII")
def test_pii_never_leaks_to_answer_or_log():
    """PII in the corpus must not appear in answers or logs."""
    assert True

@pytest.mark.skip(reason="implement with tracing")
def test_trace_covers_every_step():
    """Every agent step and tool call must appear in the audit trail."""
    assert True

@pytest.mark.skip(reason="implement with reproducibility")
def test_rerun_reproduces_metrics():
    """A seeded re-run reproduces reported metrics within tolerance."""
    assert True

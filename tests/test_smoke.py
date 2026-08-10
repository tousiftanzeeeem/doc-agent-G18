"""End-to-end tiny run. Passes once students implement the stages."""
import pytest
from doc_agent import config, pipeline

@pytest.mark.skip(reason="enable after implementing stages")
def test_answer_is_grounded_and_cited():
    ans = pipeline.answer("sample question", config.load())
    assert ans.grounded and len(ans.citations) >= 1

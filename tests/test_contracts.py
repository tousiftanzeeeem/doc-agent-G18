from doc_agent.contracts import Answer, Citation

def test_answer_contract():
    a = Answer(text="x", citations=[Citation(chunk_id="c1", span=(0, 1))],
               grounded=True, confidence=0.9)
    assert a.grounded and a.citations[0].chunk_id == "c1"

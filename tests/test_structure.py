"""STRUCTURE LOCK — CI fails if required modules/callables/signatures drift."""
import importlib, inspect

REQUIRED = {
    "doc_agent.pipeline": ["build_knowledge_base", "answer"],
    "doc_agent.contracts": ["Page", "Region", "Chunk", "Query", "Answer", "ToolResult"],
    "doc_agent.ingest.enhance": ["Enhancer", "run"],
    "doc_agent.vision.layout": ["detect"],
    "doc_agent.vision.ocr": ["Reader", "transcribe"],
    "doc_agent.index.store": ["build", "load"],
    "doc_agent.retrieval.retriever": ["Retriever"],
    "doc_agent.agent.agent": ["Agent"],
    "doc_agent.agent.tools": ["REGISTRY"],
    "doc_agent.agent.guardrails": ["Guardrails"],
    "doc_agent.agent.hitl": ["escalate"],
    "doc_agent.rl.rlvr": ["verifiable_reward"],
    "doc_agent.rl.train_rl": ["train"],
    "doc_agent.eval.metrics": ["recall_at_k", "groundedness"],
    "doc_agent.serve.api": ["app"],
    "doc_agent.mlops.tracking": ["init_run"],
}

def test_required_symbols_exist():
    for mod, names in REQUIRED.items():
        m = importlib.import_module(mod)
        for n in names:
            assert hasattr(m, n), f"missing {mod}.{n} — do not remove/rename"

def test_agent_loop_signature():
    from doc_agent.agent.agent import Agent
    assert set(inspect.signature(Agent.run).parameters) == {"self", "query_text"}

def test_tool_names_locked():
    from doc_agent.agent.tools import REGISTRY
    got = {t.name for t in REGISTRY}
    assert got == {"retrieve","rerank","read_page","enhance_page","extract",
                   "aggregate","cite","calculator","escalate_to_human"}
REQUIRED_V2 = {
    "doc_agent.settings": ["settings"],
    "doc_agent.logging_conf": ["get_logger"],
    "doc_agent.llm.client": ["LLM"],
    "doc_agent.llm.prompts": ["SYNTHESIZE"],
    "doc_agent.llm.postprocess": ["format_answer"],
    "doc_agent.agent.hitl_store": ["enqueue", "pending"],
    "doc_agent.training.adapt": ["apply_lora", "quantize"],
    "doc_agent.eval.calibration": ["ece"],
    "doc_agent.governance.pii": ["detect", "redact"],
    "doc_agent.data.versioning": ["snapshot"],
    "doc_agent.data.validate": ["validate"],
}

def test_required_v2_symbols_exist():
    import importlib
    for mod, names in REQUIRED_V2.items():
        m = importlib.import_module(mod)
        for n in names:
            assert hasattr(m, n), f"missing {mod}.{n} — mandatory home"


REQUIRED_XCUT = {
    "doc_agent.hooks": ["register", "run", "SEAMS"],
    "doc_agent.wiring": ["register_all"],
    "doc_agent.governance.pii": ["register"],
    "doc_agent.agent.guardrails": ["register"],
    "doc_agent.llm.postprocess": ["register"],
    "doc_agent.logging_conf": ["register"],
}

def test_crosscutting_wiring_exists():
    import importlib
    for mod, names in REQUIRED_XCUT.items():
        m = importlib.import_module(mod)
        for n in names:
            assert hasattr(m, n), f"missing {mod}.{n} - cross-cutting wiring"

def test_seams_are_locked():
    from doc_agent import hooks
    assert set(hooks.SEAMS) == {"after_ingest","after_ocr","before_index","after_retrieve",
        "on_step","on_tool_call","before_answer","after_answer","on_log"}

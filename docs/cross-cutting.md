# Cross-cutting features — audit & touchpoint map

Not every feature lives in one file. **Vertical** features are a pipeline stage and own one module.
**Horizontal** features are a property of the whole system: they have ONE owner (the logic) but run at
MANY touchpoints. The clean tree holds because horizontals attach at fixed **hook seams** (`hooks.py`),
wired in one manifest (`wiring.py`) - so their touchpoints are explicit, not scattered.

## Vertical features (own one module - the clean tree is enough)
ingest/loader - preprocess - enhance - vision/layout - vision/ocr - index/chunk - embed - store -
retrieval/retriever - rerank - agent loop - memory - llm/client - prompts - rl/* - training/* - adapt -
serve/api - ui - mlops/registry - eval/metrics (each is one transform in one place).

## Horizontal features (owner + touchpoints)

### Runtime cross-cutting - attach via hook seams
| Feature | Owner (logic) | Seams (touchpoints) | Enforced by |
|---|---|---|---|
| No-hallucination / grounding | `llm/postprocess.register` | BEFORE_ANSWER (+ prompt in `llm/prompts`, metric in `eval/metrics`) | `tests/test_crosscutting.py::grounding` |
| Security / prompt-injection | `agent/guardrails.register` | ON_TOOL_CALL (+ tool least-privilege in `agent/tools`, API auth in `optional/api_security`) | `...::injection` |
| PII / privacy | `governance/pii.register` | AFTER_OCR, BEFORE_ANSWER, ON_LOG | `...::pii_never_leaks` |
| Auditability / tracing | `logging_conf.register` | ON_STEP, ON_TOOL_CALL, AFTER_ANSWER | `...::trace_complete` |

### Property cross-cutting - satisfied across files, verified by a test (no runtime hook)
| Feature | Touchpoints (all must hold) | Enforced by |
|---|---|---|
| Reproducibility | `scripts/set_seed` + `requirements.lock` + deterministic ops in `training/*` + `data/versioning` + `optional/config_snapshot` | `...::reproducible` |
| Calibration | fit/measure in `eval/calibration` + attach confidence in `llm/postprocess` | `...::calibrated` |
| Fairness | subgroup metadata emitted in `ingest` + audit in `eval/fairness` | `...::fairness_reported` |
| Explainability | scores exposed by `retrieval` + attention by `vision` + trace by `logging` + consumed in `eval/interpret` | `...::explanation_available` |
| Scalability | index type `index/store` + batching `serve` + `optional/cache` + `optional/stream_ingest` | `...::scales` (benchmark) |

## The rule
- A horizontal feature's **logic** lives in its owner module.
- Its **wiring** is one line per seam in the owner's `register()`, listed in `wiring.py`.
- Its **completeness** is proven by a cross-cutting test, not by "the file exists".
- Do not inline cross-cutting logic into stage files; attach it at a seam.

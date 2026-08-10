# doc-agent — Codebase Guide

*Part 1: what every folder, file, and function does. Part 2: the order to build them in. `[fill]` = you implement · ⭐ mandatory · ➕ optional.*

**Cross-cutting features** (grounding, security, PII, tracing) don't live in one file: their logic sits in an owner module and they run at fixed **seams** via `hooks.py`, wired in `wiring.py`. See `docs/cross-cutting.md` for the full owner→touchpoint map.

---

# Part 1 · What each part does

### `configs/` — Set the task and pick models/parameters. The only place you configure the project.
- **`config.yaml`** — Model and hyperparameter choices per stage.
- **`design_choices.md`** — Design-choice table to fill for A2.
- **`task.yaml`** — Your problem spec.

### `data/` — Where the scanned corpus lives; states the data contract and splits.
- **`README.md`** — Data contract.

### `src/doc_agent/` — The package. Fixed structure; you fill the marked function bodies only.
- **`config.py`** — FIXED config loader
    - `load(path)` — to implement
    - `load_task(path)` — to implement
- **`contracts.py`** — FIXED data contracts. Do not change fields or types
    - `Page` (class)
    - `Region` (class)
    - `Chunk` (class)
    - `Query` (class)
    - `Citation` (class)
    - `Answer` (class)
    - `ToolResult` (class)
- **`hooks.py`** — FIXED — cross-cutting seam. Horizontal features register handlers here; the pipeline and agent
    - `register(seam, handler)` — Attach a handler to a seam. Called by each feature's register() via wiring.py
    - `run(seam, ctx)` — Run every handler registered at `seam`, threading ctx through. Called at fixed points only
    - `clear()` — to implement
- **`logging_conf.py`** — FIXED — structured logging (auditable NFR). Use get_logger(), never print()
    - `get_logger(name)` — to implement
    - `register(hooks)` — Wire structured tracing at each seam (auditable trail)
- **`pipeline.py`** — FIXED end-to-end order (Stages 0-9) + cross-cutting seams
    - `build_knowledge_base(cfg)` — to implement
    - `answer(query_text, cfg)` — to implement
- **`settings.py`** — FIXED — typed settings from environment (secrets live here, never in code/config)
    - `Settings` (class)
- **`wiring.py`** — FIXED — the single auditable manifest of cross-cutting features and where they attach
    - `register_all(cfg)` — to implement

### `src/doc_agent/ingest/` — Load, clean, and enhance the scanned page images.
- **`enhance.py`** — Stage 1 — ENHANCEMENT (VAE / diffusion) — generative denoise / super-resolution of degraded scans
    - `Enhancer` (class) — Model set by cfg['enhance'].  train() and apply()
        - `train(pages)` — to implement
        - `apply(pages)` — to implement
    - `run(pages, cfg)` — to implement
- **`loader.py`** — Stage 1 — load scanned page-images
    - `load_pages(cfg)` — Read data/raw/ -> list[Page]
- **`preprocess.py`** — Stage 1 — deskew / denoise / binarize / augment
    - `run(pages, cfg)` — Classical preprocessing

### `src/doc_agent/vision/` — Detect page layout, then read the text (OCR).
- **`layout.py`** — Stage 2 — layout detection / segmentation
    - `detect(pages, cfg)` — Detect text/table/figure/heading regions
- **`ocr.py`** — Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)
    - `Reader` (class) — Model set by cfg['ocr']. Baseline: pretrained TrOCR/Donut/Tesseract
        - `transcribe_region(region)` — to implement
    - `transcribe(regions, cfg)` — Regions -> text chunks.  (calls Reader)

### `src/doc_agent/index/` — Chunk the text, embed it, and store it in a vector index.
- **`chunk.py`** — Stage 4 — chunk text
    - `split(chunks, cfg)` — Re-chunk to cfg['index'] size/overlap
- **`embed.py`** — Stage 4 — embed chunks
    - `encode(chunks, cfg)` — Embed with cfg['embed']['model']
- **`store.py`** — Stage 4 — vector store
    - `build(chunks, vectors, cfg)` — Persist a vector index (cfg['index']['type'])
    - `load(cfg)` — to implement

### `src/doc_agent/retrieval/` — Find the most relevant chunks for a query.
- **`rerank.py`** — Stage 5 — reranking
    - `rerank(query, candidates, cfg)` — Cross-encoder rerank if cfg['retrieve']['rerank']
- **`retriever.py`** — Stage 5 — dense retrieval
    - `Retriever` (class)
        - `retrieve(query, k)` — Top-k dense retrieval

### `src/doc_agent/llm/` — All language-model calls, prompts, and answer formatting.
- **`client.py`** — LLM — the single LLM call wrapper (all model calls go through here)
    - `LLM` (class) — Model set by cfg['agent']. Key from settings.  complete()
        - `complete(prompt)` — to implement
- **`postprocess.py`** — LLM — answer post-process / format / abstention
    - `format_answer(raw, citations)` — Attach citations, set grounded/confidence, enforce abstention
    - `register(hooks)` — Wire the grounding / abstention gate.  (abstain if answer unsupported by evidence)
- **`prompts.py`** — LLM — FIXED prompt template registry (all prompts live here)

### `src/doc_agent/agent/` — The reason→act→observe loop, its tools, memory, and human-in-the-loop.
- **`agent.py`** — Stage 6 - FIXED loop - perceive -> decide -> act -> observe, with cross-cutting seams
    - `Agent` (class) — FIXED loop. Implement decide() (the policy) and synthesize() only
        - `run(query_text)` — to implement
        - `decide(state)` — Choose next tool+args. Rule-based (baseline) or RL policy (Stage 7)
        - `act(action)` — to implement
        - `synthesize(state)` — Grounded, cited answer; abstain if unsupported (no-hallucination)
- **`guardrails.py`** — Stage 6 — SECURITY — autonomy, budgets, prompt-injection defense
    - `Guardrails` (class) — Enforce autonomy level, step/cost budget, and instruction/content isolation
        - `reset()` — to implement
        - `check(action)` — Raise if over budget / disallowed autonomy / injection detected
    - `register(hooks, cfg)` — Wire guardrails into every tool call.  (call Guardrails.check)
- **`hitl.py`** — HITL — human-in-the-loop review queue
    - `escalate(reason, context)` — Queue for human review; block action until approved
    - `review_queue()` — Return pending items for the reviewer UI
- **`hitl_store.py`** — HITL — persistent review queue (survives restarts)
    - `enqueue(item)` — Persist a pending review item; return id.  (sqlite/json)
    - `pending()` — to implement
    - `resolve(item_id, decision)` — to implement
- **`memory.py`** — Stage 6 — working/episodic memory
    - `Memory` (class)
        - `add(item)` — to implement
        - `recall(query)` — to implement
- **`tools.py`** — Stage 6 — FIXED tool interface — the agent's tools
    - `Tool` (class)
        - `__call__()` — to implement
    - `Retrieve` (class)
        - `__call__(query, k)` — to implement
    - `Rerank` (class)
        - `__call__(query, candidates)` — to implement
    - `ReadPage` (class)
        - `__call__(page_id)` — to implement
    - `EnhancePage` (class)
        - `__call__(page_id)` — to implement
    - `Extract` (class)
        - `__call__(field, chunk_id)` — to implement
    - `Aggregate` (class)
        - `__call__(op, items)` — to implement
    - `Cite` (class)
        - `__call__(chunk_id, span)` — to implement
    - `Calculator` (class)
        - `__call__(expr)` — to implement
    - `EscalateToHuman` (class)
        - `__call__(reason, context)` — to implement

### `src/doc_agent/rl/` — Train the tool-selection policy and the verifiable-reward (RLVR) reasoning.
- **`env.py`** — Stage 7 — Gymnasium env for tool/retrieval selection
    - `ToolSelectionEnv` (class) — State=agent context, Action=tool choice, Reward=task success under budget
        - `reset()` — to implement
        - `step(action)` — to implement
- **`policy.py`** — Stage 7 — policy network
    - `Policy` (class)
        - `act(state)` — to implement
- **`rlvr.py`** — Stage 7 — RLVR — verifiable reward on extraction accuracy
    - `verifiable_reward(prediction, gold)` — +1 if extraction exactly matches gold, else 0. Drives RLVR/GRPO
- **`train_rl.py`** — Stage 7 — RL/RLVR training loop
    - `train(cfg)` — Train the tool-selection policy (cfg['rl']['algo']); RLVR on fact tasks

### `src/doc_agent/training/` — Train and adapt (LoRA/quantize) the models.
- **`adapt.py`** — Stage 8 — affordable adaptation — LoRA / quantization
    - `apply_lora(model, cfg)` — Wrap a component with LoRA per cfg
    - `quantize(model, cfg)` — Post-training quantization per cfg
- **`datamodule.py`** — Training — Lightning datamodule
    - `DocDataModule` (class)
        - `setup(stage)` — to implement
- **`lit_modules.py`** — Training — Lightning modules per trainable component
    - `LitComponent` (class) — Wrap enhancer / OCR / retriever training.  training_step + configure_optimizers
        - `training_step(batch, idx)` — to implement
        - `configure_optimizers()` — to implement
- **`train.py`** — Training — unified entrypoint
    - `main(component, cfg)` — Train one component with a seeded Lightning Trainer + W&B logger

### `src/doc_agent/eval/` — Measure, ablate, stress-test, and explain the system.
- **`ablation.py`** — Stage 9 — ablation harness
    - `run(cfg)` — Toggle each stage off; report metric deltas
- **`calibration.py`** — Stage 9 — confidence calibration (calibrated-confidence NFR)
    - `temperature_scale(logits, labels)` — Fit temperature on val; return scaler
    - `ece(confidences, correct)` — to implement
- **`fairness.py`** — Stage 9 — subgroup audit
    - `audit(cfg)` — to implement
- **`interpret.py`** — Stage 9 — EXPLAINABLE — why-retrieved / where-looked
    - `explain(answer, cfg)` — Grad-CAM on read region + retrieval-score attribution
- **`judge.py`** — Stage 9 — LLM-as-judge for non-verifiable inference
    - `judge(query, answer)` — Score open-ended answers (causal/summary/intent)
- **`metrics.py`** — Stage 9 — metrics
    - `ocr_f1(pred, gold)` — to implement
    - `recall_at_k(retrieved, gold, k)` — to implement
    - `groundedness(answer)` — to implement
    - `citation_accuracy(answer)` — to implement
    - `ece(confidences, correct)` — to implement
    - `subgroup_gap(scores_by_group)` — to implement
- **`robustness.py`** — Stage 9 — OOD / scan-quality stress
    - `stress(cfg)` — to implement

### `src/doc_agent/serve/` — Serve the agent over an API and a UI.
- **`api.py`** — Stage 8 — FastAPI service
    - `answer(q)` — Return grounded, cited answer.  (calls pipeline.answer)
    - `health()` — to implement
- **`ui.py`** — Stage 8 — Gradio demo
    - `launch(cfg)` — Gradio UI over /answer

### `src/doc_agent/mlops/` — Track experiments, version models, monitor for drift.
- **`monitor.py`** — MLOps — drift / latency monitoring
    - `check_drift(cfg)` — Detect input drift (scan quality) + latency SLO breach; trigger retrain
- **`registry.py`** — MLOps — model registry
    - `register(component, path, metrics)` — Version a checkpoint + its metrics
- **`tracking.py`** — MLOps — experiment tracking (W&B)
    - `init_run(cfg, tags)` — Start a W&B run; log config
    - `log(metrics)` — to implement

### `src/doc_agent/governance/` — Detect and redact personal data (PII).
- **`pii.py`** — Governance — PII detection + redaction (mandatory)
    - `detect(text)` — Return (start,end,type) PII spans
    - `redact(text)` — to implement
    - `register(hooks)` — Wire PII redaction into the pipeline.  the handler (call redact())

### `src/doc_agent/data/` — Version the corpus and validate data quality.
- **`validate.py`** — Data — data schema/quality validation at ingest
    - `validate(pages)` — Assert min pages/words, format, no leakage across splits
- **`versioning.py`** — Data — corpus versioning (which corpus version -> which result)
    - `snapshot(corpus_dir)` — Hash + record a corpus version id.  (or wire DVC)

### `src/doc_agent/optional/` — Extra features you build only if your data speciality or NFR needs them.
- **`README.md`** — Which optional module each profile needs.
- **`api_security.py`** — OPTIONAL — API auth + rate limiting
    - `require_auth(token)` — to implement
    - `rate_limit(client)` — to implement
- **`cache.py`** — OPTIONAL — embedding/retrieval/answer cache
    - `get(key)` — to implement
    - `put(key, val)` — to implement
- **`config_snapshot.py`** — OPTIONAL — snapshot exact run config for reproducibility+
    - `snapshot(cfg, out)` — to implement
- **`dp.py`** — OPTIONAL — differential privacy (DP-SGD) for shared models
    - `make_private(optimizer, cfg)` — to implement
- **`query_expansion.py`** — OPTIONAL — query embedding + expansion
    - `expand(query, cfg)` — to implement
- **`retrain_trigger.py`** — OPTIONAL — automated retrain on drift/SLA breach
    - `maybe_retrain(metrics, cfg)` — to implement
- **`stream_ingest.py`** — OPTIONAL — batch/streaming ingestion for very large corpora
    - `stream(cfg)` — to implement
- **`synthetic_data.py`** — OPTIONAL — generate synthetic text-lines for tiny-label training
    - `generate(cfg)` — to implement

### `tests/` — Automated checks CI runs; structure and cross-cutting tests lock behaviour.
- **`test_agent.py`** — Unit test home for agent.  — CI runs these
    - `test_agent_placeholder()` — to implement
- **`test_contracts.py`** — 
    - `test_answer_contract()` — to implement
- **`test_crosscutting.py`** — Cross-cutting features must work END TO END, not just exist in one file
    - `test_grounding_unsupported_query_abstains()` — An answer with no supporting evidence must abstain, not fabricate
    - `test_injection_in_document_does_not_hijack()` — A document containing 'ignore your instructions' must not change agent behaviour
    - `test_pii_never_leaks_to_answer_or_log()` — PII in the corpus must not appear in answers or logs
    - `test_trace_covers_every_step()` — Every agent step and tool call must appear in the audit trail
    - `test_rerun_reproduces_metrics()` — A seeded re-run reproduces reported metrics within tolerance
- **`test_data.py`** — Unit test home for data validation.  — CI runs these
    - `test_data_placeholder()` — to implement
- **`test_eval.py`** — Unit test home for eval.  — CI runs these
    - `test_eval_placeholder()` — to implement
- **`test_ingest.py`** — Unit test home for ingest.  — CI runs these
    - `test_ingest_placeholder()` — to implement
- **`test_ocr.py`** — Unit test home for OCR.  — CI runs these
    - `test_ocr_placeholder()` — to implement
- **`test_retrieval.py`** — Unit test home for retrieval.  — CI runs these
    - `test_retrieval_placeholder()` — to implement
- **`test_smoke.py`** — End-to-end tiny run. Passes once students implement the stages
    - `test_answer_is_grounded_and_cited()` — to implement
- **`test_structure.py`** — STRUCTURE LOCK — CI fails if required modules/callables/signatures drift
    - `test_required_symbols_exist()` — to implement
    - `test_agent_loop_signature()` — to implement
    - `test_tool_names_locked()` — to implement
    - `test_required_v2_symbols_exist()` — to implement
    - `test_crosscutting_wiring_exists()` — to implement
    - `test_seams_are_locked()` — to implement
- **`test_tools.py`** — 
    - `test_registry_is_tool_subclasses()` — to implement

### `grading_kit/` — the one folder that makes the project reproducible & gradable
- **`manifest.yaml`** — the single entry point: your three axes + pointers to the corpus, the held-out slice, and the build/run/eval commands. A grader opens only this.
- **`heldout_pages/`** — page-images you set aside and never OCR-train on.
- **`labels.jsonl`** — ground-truth transcriptions for those pages (the oracle: OCR is scored against them, and fresh questions are authored from them).

### `grading_kit/` also holds your evaluation set (moved from `tasks/`)
- **`grading_kit/tasks.jsonl`** — your eval questions (fact + inference), one per line.
- **`grading_kit/success_check.py`** — verifies whether an answer passes a task.

### `docs/` — governance docs
- **`model_card.md`** — the model card / datasheet you complete in A4.
- **`cross-cutting.md`** — the owner→touchpoint map (read-only reference).

### `notebooks/` — exploration you edit (kept out of `src/`)
- **`eda.ipynb`** — A1 corpus exploration.
- **`kb_demo.ipynb`** — A2 OCR-quality + retrieval demo.

### `data/` — corpus + provenance you edit
- **`provenance.md`** — A1 corpus source, licence, page/word counts. (Scans go in `data/raw/`.)

### `scripts/` — run commands you edit
- **`get_data.sh`** — A1 fetch/recreate the corpus. **`build_index.sh`** — A2 build the index. **`run.sh`** — A3 reproduce all results. (`run_ingest/index/eval.py`, `set_seed.py` already provided.)

### `reports/` — write-ups and artifacts you edit
- **`pipeline_diagram.md`** (A2) · **`eval_report.md`** (A3) · **`final_report.md`**, **`brochure.md`**, **`demo_link.txt`**, **`video_link.txt`** (A4) · **`figures/`** for images.

### `traces/`, `bonus/`, `transcripts/` — drop-in folders
- **`traces/`** (A3) agent traces · **`bonus/`** (A4) bonus evidence · **`transcripts/`** each member's `<student number>.txt`.

**Rule: every file you submit already exists as a stub — you *edit* it, you never create a new file.**

You submit your **whole repository** at four tagged points (`a1-submit` … `a4-submit`) — the commit history is the submission. See **How-To-Submit** for the workflow.

---

# Part 2 · What to build, in order

*Build top to bottom. Within each assignment do the **⭐ Core** first (everyone, in build order), then — only if it applies to you — your one **▲ Data-speciality** enhancement, then your one **▲ NFR** enhancement. Build order is preserved inside each of the three groups. Every file has a stub; you **edit** it, you never create one. Each enhancement carries its **E-number from CLAUDE.md §10**, and every item in that backlog appears here under the assignment where it is built — so this list is the whole map.*

**Action tags:** `[fill]` write a function body · `[wire]` write logic **and** attach it to hook seams (see `wiring.py`) · `[author]` write config/tasks/reports/tests, not Python · `[read-only]` fixed, never edit.

**Tier tags:** ⭐ **core** — every group, always · ▲ **gated** — required only if it matches your primary NFR or data speciality · ➕ **bonus** — extra credit.

**Never edit (CI will fail you):** the agent loop in `agent/agent.py`, the stage order in `pipeline.py`, the seams in `hooks.py`, the data types in `contracts.py`, and the tool names/signatures in `agent/tools.py`.

---

### A1 — Frame & data

**⭐ Core (everyone, in build order)**
1. `[author]` **configs/task.yaml** — declare your domain, data speciality, primary NFR + target, and corpus source/licence. *(speciality and NFR are chosen here; you build them in A2–A4.)*
2. `[author]` **configs/config.yaml** — pick a model and parameters for each stage.
3. `[author]` **scripts/get_data.sh** — fetch or recreate your corpus into `data/raw/`.
4. `[author]` **data/provenance.md** — record corpus source, licence, and page/word counts.
5. `[author]` **grading_kit/manifest.yaml** — declare your axes (domain, data-speciality, NFR); point to the corpus, held-out slice, build/run commands, and `traces/run.jsonl`.
6. `[author]` **grading_kit/heldout_pages/ + labels.jsonl** — transcribe a few page-images as ground truth (grows through A2–A3; seeds E14's labels).
7. `[fill]` **data/validate.py → validate()** — enforce ≥300 pages and ≥60k words, split by document.
8. `[fill]` **data/versioning.py → snapshot()** — record a version id for the exact corpus used.
9. `[author]` **notebooks/eda.ipynb** — explore counts, scan quality, fonts *(this is your data-speciality difficulty analysis)*.
10. `[fill]` **logging_conf.py → get_logger()** — structured logging (never print).
11. `[author]` **settings.py + .env** — secrets in `.env`, read via `settings`.
12. `[author]` **transcripts/<student number>.txt** — your AI-chat transcript (one per milestone).

*A1 builds no gated enhancements — your speciality and NFR are **chosen and analysed** here, then built in later milestones.*

---

### A2 — Build the knowledge base

**⭐ Core (everyone, in build order)**
1. `[fill]` **ingest/loader.py → load_pages()** — read scanned pages into Page objects.
2. `[fill]` **ingest/preprocess.py → run()** — deskew, denoise, binarize, augment (baseline).
3. `[fill]` **vision/layout.py → detect()** — detect text, table, figure, heading regions (baseline).
4. `[fill]` **vision/ocr.py → Reader, transcribe()** — fine-tune a pretrained OCR model; transcribe regions (baseline).
5. `[fill]` **index/chunk.py → split()** — chunk text with overlap (baseline).
6. `[fill]` **index/embed.py → encode()** — embed chunks (baseline).
7. `[fill]` **index/store.py → build(), load()** — build/load the vector index (baseline).
8. `[author]` **scripts/build_index.sh** — one command that builds the index.
9. `[author]` **notebooks/kb_demo.ipynb** — show OCR quality and one retrieval example.
10. `[author]` **reports/pipeline_diagram.md** — diagram the knowledge-base pipeline.
11. `[author]` **configs/design_choices.md** — fill the design table for Stages 1–4.

**▲ Data-speciality (implement only your condition, in build order by stage)**
- **E1** — severely degraded scans / bleed-through / show-through → `ingest/preprocess.py` (+ ➕ `ingest/enhance.py`, VAE/diffusion repair).
- **E2** — table/figure-heavy or multi-column reading-order → `vision/layout.py`.
- **E3** — non-English script / font-typography diversity / old-orthography → `vision/ocr.py` (script-tuned OCR; orthography normalized in `index/chunk.py`).
- **E24** — handwritten / manuscript (HTR) → `vision/ocr.py`.
- **E25** — math / scientific notation → `vision/ocr.py`.
- **E23** — Bangla–English code-switching → `vision/ocr.py` + `index/embed.py` (language-ID + multilingual embed).
- **E26** — dirty-OCR-provided (skip OCR, clean the given text) → `index/chunk.py`.

**▲ NFRs (implement only your NFR, in build order)**
- **E4** — paragraph/semantic chunking (better retrieval) → `index/chunk.py`.
- **E5** — scalable ANN index, HNSW/IVF (+ `optional/stream_ingest.py` for huge corpora) → `index/store.py`.

---

### A3 — Retrieve, reason, evaluate

**⭐ Core (everyone, in build order)**
1. `[fill]` **retrieval/retriever.py → retrieve()** — return the top-k chunks for a query.
2. `[fill]` **llm/client.py → LLM.complete()** — one wrapper for all LLM calls.
3. `[author]` **llm/prompts.py** — the decide, synthesize, and judge prompts.
4. `[fill]` **agent/tools.py** — implement each tool (retrieve … escalate_to_human).
5. `[fill]` **agent/memory.py → recall()** — store/recall context across steps.
6. `[fill]` **agent/agent.py → decide(), act(), synthesize()** — the policy + a grounded, cited, abstaining answer; includes the **E8** agent re-search loop and **E9** abstention (the faithful baseline). **decide() must implement evidence-gated re-search (the mandatory agentic behaviour):** if `top_score < weak_threshold`, **widen `k` by `k_step` and retrieve again**; keep widening until it recovers, and **abstain once `k` would exceed `k_max`**. Homes: `Chunk.score` (contracts), the `k_step/k_max/weak_threshold` knobs (config.yaml), `is_weak()/next_k()` (retriever.py). Graded fail-closed by the A3 gate; a fixed retrieve→answer path is not agentic. *(loop + seams fixed.)*
7. `[wire]` **llm/postprocess.py → format_answer(), register()** — citations, confidence, abstention (**E9**); wire the grounding gate to BEFORE_ANSWER.
8. `[wire]` **logging_conf.py → register()** — **E12** tracing to ON_STEP, ON_TOOL_CALL, AFTER_ANSWER; **emit `traces/run.jsonl`** (one `contracts.TraceStep` per step) so the agentic-feature check can read the trajectory.
9. `[fill]` **training/ (datamodule, lit_modules, train)** — train the components you fine-tune.
10. `[fill]` **eval/metrics.py** — **E14** OCR-F1, recall@k, groundedness, citation accuracy, subgroup gap (against `grading_kit/labels.jsonl`).
11. `[fill]` **eval/ablation.py** — **E16** ablation study (everyone does this).
12. `[author]` **grading_kit/tasks.jsonl + success_check.py** — your eval questions and the checker; **include ≥ 1 `{needs_research:true, needs_research:false}` task pair** (the A3 agentic gate runs these to see re-search fire).
13. `[author]` **scripts/run.sh + reports/eval_report.md** — one command to reproduce; write metrics, ablation, NFR result, and **E18** baseline comparison (figures in `reports/figures/`).
14. `[author]` **traces/** — at least one full agent trace.
15. `[author]` **tests/test_*.py + test_crosscutting.py** — per-stage + cross-cutting tests; plus the engineering hygiene everyone keeps green: **E37** unit tests, **E36** type hints vs `contracts.py`, **E38** ruff + black, **E39** green CI.

*No data-speciality step here — your data condition was handled in A2's knowledge base.*

**▲ NFRs (implement only your NFR, in build order)**
- **E6** — reranking, for precision-first / recall-first → `retrieval/rerank.py`.
- **verify-and-correct** — re-check grounding and re-synthesize if unsupported, for precision-first → `agent/agent.py`, `llm/postprocess.py`.
- **E7** — hybrid dense + BM25 retrieval, for recall-first (+ `optional/query_expansion.py`) → `retrieval/retriever.py`.
- **multi-hop** — chain retrievals (2nd query built from the 1st observation), for recall-first → `agent/agent.py` (decide()).
- **E28 / E29** — memory-efficient / compute-budget: LoRA + quantize to run on Colab → `training/adapt.py`.
- **E31** — data-efficient, quality per labelled page (+ `optional/synthetic_data.py`) → `training/`, `eval/`.
- **E17** — calibrated confidence, ECE ≤ 0.05 → `eval/calibration.py`.
- **E32** — explainable, attribution ≥ 0.90 → `eval/interpret.py`.
- **E33** — fair / unbiased, subgroup gap ≤ 0.05 → `eval/fairness.py`.
- **E34** — robust / reliable, OOD F1 drop ≤ 10% → `eval/robustness.py`.
- NFR strictenings of the core baselines: **secure** — injection ASR ≤ 5% (`eval/robustness.py` + `agent/guardrails.py`; + `optional/api_security.py`); **private** — PII recall ≥ 0.98 (`governance/pii.py`; + `optional/dp.py`); **auditable** — trace = 100% (`mlops/tracking.py`; + `optional/config_snapshot.py`).

**➕ Bonus**
- **runtime tool-routing** — pick the tool from the question/observation type (calc→calculator, table→aggregate) → `agent/agent.py` (decide()).
- **E22** — RL / RLVR: train tool-selection, reward correct verifiable extractions → `rl/ (env, policy, rlvr, train_rl)`.
- **E15** — LLM-judge: score open-ended answers (only if you have judged questions) → `eval/judge.py`.
- **E13** — human-in-the-loop: optional escalation to a person → `agent/hitl.py`, `hitl_store.py`.

---

### A4 — Ship & govern

**⭐ Core (everyone, in build order)**
1. `[fill]` **serve/api.py → /answer** — serve grounded answers over HTTP.
2. `[fill]` **serve/ui.py → launch()** — launch the Gradio demo (**E19** serving polish: UI + `Dockerfile`).
3. `[fill]` **mlops/tracking.py** — track runs (core; the **E20** governance backbone).
4. `[author]` **docs/model_card.md** — **E20** model card / datasheet.
5. `[author]` **reports/final_report.md** — real-world, safety, and NFR results.
6. `[author]` **reports/brochure.md** — the one-page brochure (export to `brochure.pdf`).
7. `[author]` **reports/demo_link.txt + video_link.txt** — live demo and video URLs.
8. `[author]` **tests/test_smoke.py** — un-skip; end-to-end returns a grounded, cited answer (**E39** CI stays green).

*No data-speciality step here — see A2.*

**▲ NFRs (implement only your NFR, in build order)**
- **E21** — offline / portable: no query-time network + size budget → `serve/api.py`, `eval/`.
- **E27** — low-latency, p95 (+ `optional/cache.py`) → `serve/`, `mlops/monitor.py`.
- **E30** — cost-efficient, $/1k queries → `mlops/monitor.py`.
- **E35** — accessible: audio / screen-reader → `serve/ui.py`.
- reproducible / drift governance: **E20** model **registry** (reproducible NFR) and **monitor** drift (scalable / drift NFR; + `optional/retrain_trigger.py`) → `mlops/registry.py`, `mlops/monitor.py`.

**➕ Bonus**
- `bonus/` — evidence for handwriting / real client / multi-agent (if attempted).

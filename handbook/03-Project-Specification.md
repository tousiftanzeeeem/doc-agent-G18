# Project Specification — The Document-Reading Agent
### A scanned-document Agentic-RAG substrate: one pipeline, one domain per group, the whole course built as a working system

*Working draft. Everything here is written as **mandatory** so we can see the full shape; you'll then decide
what to relax. Each pipeline stage names the course classes (C1–C36) it exercises, so the project is a
front-to-back re-derivation of the syllabus — a student who builds the whole thing has *built* the course,
not just heard it.*

---

## Part A — The idea

Every team builds the **same kind of system**: an **agent that reads a corpus of scanned documents, retrieves
the relevant pages, and answers questions grounded in them, with citations.** The pipeline is identical for
everyone; what differs is the **document collection**, which sets the domain (history, botany, law, medicine,
cooking, engineering…). One substrate to standardize, teach, and grade; one unique corpus per group to make
the work genuinely their own (see Part G).

**Differentiation across groups** rides three independent axes, so one substrate never means one project:
**domain × data speciality × primary NFR** — detailed in Part G and
Part G, and operationalized in the supporting references **Data-Speciality-Catalog**, **Domains-And-Natural-Fit**, and
the workbook's **NFR** sheet.

The design choice that makes this a *whole-course* project rather than a chatbot: the knowledge base is
**scanned/handwritten documents — images before they are text.** Nothing is retrievable until the vision stack
works. So the vision pillar (CNN lineage, detection/segmentation, attention-based OCR, generation for
enhancement) isn't a bolt-on — it's the *front of the pipeline the whole system depends on*. Vision and
language are taught as **one system**, because here they literally are one system.

**Prime directive for the student:** raw page image → read it → find it → answer from it, on the leash the
agent standard defines. Build that, honestly, on your own corpus.

**RAG agent — tasks & tools (build all).**
Tasks: parse query → plan → retrieve → (optionally) enhance/re-OCR a page → read → synthesize grounded answer
with citations → self-check → abstain if unsupported.
Tools: `retrieve(query,k)` (vector search), `rerank(candidates)`, `read_page(id)` (OCR-on-demand),
`enhance_page(id)` (denoise/super-res), `extract(field)` (fact extraction), `aggregate(op,items)`
(count/sum/etc.), `cite(span)`, `calculator`, `escalate_to_human()` (HITL). Fixed interface; students choose
the model behind each.

**Minimum corpus (per group):** **≥ 300 pages AND ≥ 60,000 words** of usable extracted text. *Reason:* below
this, retrieval is trivial and aggregate/multi-hop tasks have too few facts to be real. The *huge-corpus*
the *scalable* NFR / huge-corpus speciality pushes this to **≥ 100k pages**.

---

## Part B — Data

**Default source — scanned public-domain books (Internet Archive / HathiTrust).** These are *page-image* scans
(genuinely images-before-extraction), public-domain (clean licensing), and available across every subject —
so each group picks a **distinct subject domain** and downloads a scanned-book corpus in it.

> *Note on Gutenberg:* Project Gutenberg is mostly **already-transcribed clean text** — it would skip the
> entire vision pillar. Use it only as a *reference transcription* to help build an OCR eval set, not as the
> primary corpus. The **scanned page images** must come from Internet Archive / HathiTrust / similar.

**Real-world track (allowed):** a team may bring its own scanned corpus — archival records, forms, receipts,
lab notebooks, historical manuscripts — if they want a live problem. Same pipeline; they own the licensing.

**Difficulty tiers (the knob that protects the back half of the course):**
- **Baseline:** printed scanned books (tractable OCR) so every team reaches the agent layer.
- **Hard mode (opt-in):** handwritten / historical manuscripts (HTR is genuinely hard).
- **OCR policy:** a strong pretrained recognizer (Tesseract / TrOCR / Donut) may be used as the *starting
  point* — students still build layout analysis, fine-tuning, and error analysis on their corpus, but are not
  blocked spending a semester getting transcription to work from scratch.

**Uniqueness rule:** no two groups may use the same subject domain / corpus. Domains are registered
first-come or assigned (see Part G — this is the linchpin of the anti-copying design).

*Classes exercised by the data work:* **C3** (honest measurement — split by document, not page; leakage = the
same scan in two splits), **C15** (data as the lever — curation, deduplication of scans, corpus construction).

---

## Part C — The pipeline, stage by stage, mapped to the course

Each stage lists what the student builds and the classes it makes concrete. Together they cover C1–C36.

### Stage 0 — Frame the problem & the corpus
Define the task and success metric; choose the domain and obtain the scanned corpus; document licensing;
build the retrieval/extraction evaluation set. Establish the agent-on-paper: what it perceives, decides, acts
on, and the autonomy/budget/guardrail limits it runs under.
→ **C1** (the learning agent — perceive/decide/act, autonomy, guardrails, prompt injection), **C3** (metric,
baseline, leakage-safe splits), **C15** (corpus curation).

### Stage 1 — Ingest & preprocess the page images  *(← the enhancement / diffusion stage)*
Load scanned pages; deskew, denoise, binarize; augment. **Generative enhancement (this is where it lives):**
train a denoiser / super-resolver (VAE or diffusion) to clean degraded scans before recognition.
→ **C2** (the machine; augmentation as regularization), **C4** (convolution — the operations that clean an
image), **C18–C19** (generation I & II — VAE / diffusion for document-image denoising and enhancement).

### Stage 2 — Layout analysis (detection & segmentation)
Detect and segment the page into regions — text blocks, headings, figures, tables, columns, marginalia —
so recognition runs on the right pieces in the right order. Optionally model layout as a graph.
→ **C5** (CNN lineage — the detection backbone), **C16** (ViT for documents; layout-as-graph GNN — the one
place structured/permutation-aware modeling is native), **C17** (dense prediction & detection — *the core of
this stage*: segmentation, region proposals, set-prediction).

### Stage 3 — Text recognition (OCR / HTR)
Transcribe each region to text. This is the classic **CNN → attention → decoder** arc: a vision encoder reads
pixels, a sequence model aligns them to characters (CTC or attention), a decoder emits text. Fine-tune a
pretrained recognizer on the corpus.
→ **C6–C7** (trainable depth — training the reader; optimizer, LR schedule, init), **C8–C9** (memory over
time; the encoder-decoder bottleneck; CTC/attention alignment), **C10–C11** (self-attention & the Transformer
block — modern OCR/HTR, e.g. TrOCR-style).

### Stage 4 — Represent & index the text
Tokenize the extracted text; turn passages into embeddings; build the vector index. Train/adapt the
retriever's embedding model with contrastive / masked objectives.
→ **C12** (tokens & word vectors; subword tokenization), **C13** (meaning as geometry — the retriever *is*
embedding geometry; "everything becomes an embedding," including the page), **C20** (representations without
labels — contrastive / masked-image modeling to train the encoder).

### Stage 5 — Retrieval
Query → embed → retrieve top-k passages from the vector store. This is the mandatory **LLM-app / vector-DB**
component. Tune the retriever; measure recall@k on the eval set.
→ **C13** (embedding geometry), **C14** (scale & cost — the price tag of the backbone; FLOPs/memory on Colab),
**C25** (RAG — grounding the answer in retrieved text).

### Stage 6 — The agent: reason, act, answer (grounded)
The agent loops over the store: perceive the query, decide which tool to call (retrieve, enhance a page,
multi-hop follow-up), act through the harness, and answer **grounded with citations** — resisting
hallucination. Enforce autonomy levels, budgets, approvals, and prompt-injection defenses (adversarial text
inside the documents is a *real* threat here).

**Agentic behaviour is mandatory and graded, not assumed.** A fixed retrieve→answer pipeline in an agent
loop is *not* agentic. The ONE required dynamic behaviour — **evidence-gated re-search** (when the top retrieval
score is below a threshold, widen `k` and retrieve again; abstain once `k` hits `k_max`) — is what every group must show. The agent emits a
machine-readable trajectory (`traces/run.jsonl`), the group authors a matched `{needs_research true/false}`
task pair, and the **A3 agentic gate runs the agent and verifies re-search actually fires only when needed**
(on the trigger, `k` widens and the run then recovers or abstains at `k_max`; on the control, a single pass) — **fail-closed**: a
deterministic path caps the grade. Richer dynamic behaviours are optional: **multi-hop** counts toward the
recall NFR, **verify-and-correct** toward the precision NFR, and **runtime tool-routing** is bonus.
→ **C21** (multimodal — document VQA over image + text), **C24** (alignment — using an aligned answerer),
**C25** (talker-to-doer — harness, tool-use-as-text, structured output, hallucination→grounding), **C26**
(the agent loop; memory; MCP; autonomy), **C27** (trustworthy agents — prompt injection from document
content, sandboxing, context engineering), **C28** (reasoning — multi-hop chain-of-thought over pages).

### Stage 7 — Learned control & trained-in reasoning (RL + RLVR)
Two mandatory learning-to-act components: **(a)** an RL / bandit **policy that chooses tools or retrieval
actions** to maximize task success under budget; **(b)** **RLVR on verifiable extraction accuracy** — reinforce
the reasoning traces that produce correct, checkable extractions (a clean verifiable reward this domain
gives for free).
→ **C21–C22** (learning to act — value/policy, DQN, exploration), **C23** (planning — multi-hop retrieval as
search), **C29** (reasoning trained into the weights — RLVR/GRPO on extraction correctness), **C35**
(agentic RL — training the whole multi-step document-QA trajectory).

### Stage 8 — Make it affordable & serve it
Adapt cheaply (LoRA the reader/retriever), compress (quantize) for Colab-scale training and serving; serve
the pipeline behind a UI.
→ **C30** (LoRA / PEFT, MoE, distillation, quantization, pruning), **C31** (serving under real constraints —
KV-cache, batching; FastAPI + Gradio + Docker + HF Spaces).

### Stage 9 — Evaluate, harden, and account for it
Measure the whole system (transcription F1, retrieval recall@k, grounded-answer quality via LLM-as-judge,
stage-wise ablations). Harden against the messy real world — scan-quality OOD, calibration, PII. Open the
box — Grad-CAM on read regions, "why was this retrieved," fairness across scripts/languages. Then the
principle scan.
→ **C32** (knowing it works — eval design, ablation, LLM-as-judge, contamination), **C33** (robustness &
the drifting world — scan-quality OOD, calibration, differential privacy for PII in documents), **C34**
(interpretability & governance — attention/Grad-CAM, fairness, model cards, PII governance), **C36** (the
through-line — the principle scan over what they built).

---

## Part D — The components (with tiers)

Listed here as the full system so we can see the whole. **The tier system now governs which are required of
every group vs. gated vs. bonus** (see `codebase-guide.md` Part 2): **⭐ core** (all groups) · **▲ NFR/
profile-gated** (required only if the group's primary NFR or data speciality triggers it) · **➕ bonus**
(extra credit). Enforced the way the agent layer already is — an interface + a task-suite check + CI, not a
grader's goodwill. Tier tags below reflect the current mandatory/elective line; the assignment forms follow
these tiers.

1. **⭐ Vision → OCR spine:** preprocessing, layout detection/segmentation, text recognition. *(Forces C4–C11,
   C16–C17.)*
2. **➕ Generative enhancement:** a trained denoiser / super-resolver (VAE or diffusion). *(Forces C18–C19.)*
3. **⭐ Representation & retrieval:** embeddings + vector store + a query path that can only be solved by
   retrieving, not from the brain alone. *(Forces C12–C13, C20, C25 — this is the mandatory LLM-app layer.)*
4. **⭐ The grounded agent:** perceive→decide→act loop, tools, citations, guardrails, prompt-injection defense.
   *(Forces C1, C21, C24–C28.)*
5. **➕ Learned control (RL) + trained-in reasoning (RLVR):** a tool/retrieval-selection policy and a
   verifiable-reward extraction sub-task. *(Forces C21–C23, C29, C35.)*
6. **⭐ serving · ▲ efficiency (LoRA/quantize, deployment-gated):** LoRA + quantization + a public demo. *(Forces C30–C31.)*
7. **⭐ PII/fairness · ▲ calibration/interpretability (NFR-gated):** ablations, OOD on scan quality, calibration, interpretability, PII/fairness.
   *(Forces C32–C34.)*
8. **⭐ The principle scan:** map ≥8 build decisions to the 30 principles. *(Forces C36.)*

---

## Part E — 36-class coverage map

Every class is **built**, not just lectured. `native` = the pipeline forces it; `component` = a required
sub-task the spec mandates; `📖` = a sub-concept that stays a lecture exemplar in any semester project.

| Class | Built in | How |
|---|---|---|
| 1 primer · 2 machine · 3 measurement | native | the agent; every trained stage; doc-level splits |
| 4 conv · 5 CNN lineage · 6 depth · 7 why | native (7 has 📖) | preprocessing + reader backbone + training craft |
| 8 memory · 9 bottleneck · 10 self-attn · 11 block | native | OCR/HTR: CNN→attention→decoder |
| 12 tokens · 13 geometry | native | extracted text → embeddings → retrieval |
| 14 scale | native (📖 scaling-laws) | pretrained backbones; cost/FLOPs on Colab |
| 15 data · 16 bias/structured · 17 detection | native | corpus curation; ViT/GNN layout; **layout detection** |
| 18 gen-I · 19 gen-II | component | VAE / diffusion document enhancement |
| 20 SSL · 21 multimodal | native | contrastive retriever; document VQA |
| 22 act · 23 planning | component | RL tool-selection; multi-hop retrieval as search |
| 24 alignment | native | the aligned grounded answerer |
| 25 talker→doer · 26 agent loop · 27 trustworthy | native | RAG, harness, autonomy, **prompt injection from docs** |
| 28 reasoning scaffolded | native | multi-hop CoT over pages |
| 29 reasoning trained-in | component | **RLVR on verifiable extraction accuracy** |
| 30 affordable · 31 serving | native | LoRA/quantize; Gradio/Docker/HF Spaces |
| 32 eval · 33 robustness · 34 interp+gov | native | F1/recall@k/LLM-judge; scan OOD; PII; Grad-CAM |
| 35 frontier | component (📖 VLA) | agentic RL over the doc-QA trajectory |
| 36 through-line | native | the principle scan |

**Result:** 26 classes native, 10 via a mandated component, 0 whole classes unreachable. Every 📖 is a
sub-concept no undergraduate builds in a semester regardless of project (Chinchilla-scale training,
distributed training, self-play, AlphaFold, VLA).

---

## Part F — Binds to the standard stack

The substrate uses the mandated stack unchanged: **PyTorch + Lightning** (reader, retriever, RL policy),
**HuggingFace / timm** (pretrained OCR, embeddings, backbones), **torchmetrics** (F1, recall@k, calibration),
**a vector store** (the new mandatory piece — added to `src/` as `vectorstore.py` + a `retrieve()` tool in
`tools.py`), **W&B** (every run), **Gymnasium** (the RL tool-selection environment), **FastAPI + Gradio +
Docker + HF Spaces** (the demo), enforced by **pytest + GitHub Actions**. The repo's `agent.py` /
`tools.py` / `tasks.jsonl` are exactly the agent standard; the smoke test gains one line — *the retrieval
tool returns grounded context for one query.*

---

## Part G — Ensuring original work (the anti-copying design)

You raised the key risk: one common substrate invites copying between groups and free-riding within them.
The answer is **structural, not forensic** — the design is built so copied work doesn't function and each
person's contribution is recorded. Four layers, strongest first.

### Layer 1 — Copying doesn't work, because every group's task is unique
This is the linchpin, and it comes free from **three independent differentiation axes** (see
the **Data-Speciality-Catalog** reference and the workbook's **NFR** sheet). A group's fingerprint is
**domain × data speciality × primary NFR** — so no two groups solve the same
problem:
- **Domain** — a practical, real-user subject (community health, agriculture, legal aid, land records…), one per
  group, enforced by a registry. A huge
  fraction of the real work — cleaning *these* scans, *this* layout's quirks, *this* domain's vocabulary,
  *this* corpus's eval set, error analysis on *these* failures — is **corpus-specific and does not transfer.**
- **Data speciality** — the one hard corpus property (Bangla, degraded scans, multi-column, handwritten, …).
  A copied pipeline inherits the wrong speciality and won't fit another group's corpus condition.
- **Primary NFR + target** — the one quality each group optimises above baseline, with a concrete target
  (latency ≤ 300 ms, PII recall ≥ 0.98, ECE ≤ 0.05, …). Lifted code tuned for another NFR misses the target.
  (Faithful and accurate are **baseline for everyone**, not differentiators.)
- **Student-authored task suite.** `grading_kit/tasks.jsonl` + `success_check.py` are written by each group against
  *their* documents. They can't be copied because they encode domain facts only that corpus contains.
- **Corpus-specific evaluation and error analysis.** Graded artifacts (recall@k on your eval set, failure
  cases on your worst scans, the domain bias audit) can't be faked for documents you didn't process.

### Layer 2 — Provenance is recorded (who built what)
- **Per-member Git history** (already standard): feature branches, small frequent commits, PR workflow — the
  history *is* the evidence of the contribution split, and it's per person.
- **Per-member AI-chat transcripts** (already standard): each member submits their own, graded on genuine
  partnership — where they pushed back on and corrected the AI, not where it wrote their code.
- **Role rotation across milestones** (already standard): everyone touches data, modeling, and deployment, so
  no one can hide as "the person who didn't do the ML."

### Layer 3 — Graders verify authenticity, cheaply
- **Reproduce-from-corpus:** a green CI run trains/evaluates *their* pipeline on *their* declared corpus; the
  reported numbers must reproduce. A model lifted from another group won't match this corpus.
- **Held-out pages from their own domain:** the grader withholds a few documents/pages from the group's
  corpus and tests extraction/retrieval on them — trivial for a team that built it, impossible to fake.
- **Live demo + individual viva:** each member explains, and answers questions about, a design decision —
  ideally one they did *not* primarily implement (proves the whole team understands the whole system).

### Layer 4 — Detect the residue
- **Code-similarity across submissions** (MOSS-style). Because the **template/scaffold is provided to everyone
  by design**, expected overlap is *only* the scaffold; similarity in the corpus-specific `src/` beyond the
  scaffold is the signal to investigate. (Grade the *delta* over the template, which is unique by construction.)
- **Corpus/domain uniqueness registry** with a short novelty declaration per group.

**The elegant part:** the same decision that makes this project *standardizable* (a common substrate) is what
makes it *copy-resistant* — because you pair the common substrate with a **unique corpus**, and the graded
work lives in the corpus-specific delta. Standardization and originality stop being in tension.

---

## Part G2 — Questions the agent must handle (general), and the eval requirement

The system is **general**: it answers whatever questions its real user asks — factual, procedural,
comparative, safety-critical. We do **not** restrict a group to a fixed "question type"; a legal-aid tool and a
farmer's manual naturally invite different questions, and the mix follows from the **domain and the user**
(named in A1's "Who would use this"), not from an assigned axis.

What *is* required — of every group — is that the **evaluation suite makes objective grading possible**:

- **Enough verifiable questions.** A share of `tasks.jsonl` must have a single checkable answer (a name, number,
  date, or exact quote) whose gold is read from the group's labelled pages. These anchor mechanical grading
  **and** the optional RLVR reward (Stage 7): the verifiable items give a clean, automatic reward that sharpens
  a *general* agent — they do not narrow what it answers.
- **Judged questions where needed.** Non-verifiable questions (summary, causal, intent) are scored by an
  LLM-judge or human with a stated rubric — so the group *designs evaluation to its task* (C32).
- **Abstention cases.** Out-of-corpus questions whose correct answer is "insufficient evidence" — testing the
  faithful baseline.

This lives in A3's "How we authored the task suite" cell. Objective grading and RLVR therefore ride on an
**eval-suite requirement**, not on a question-type axis.

---

## Part H2 — The regulated codebase (what students receive)

Delivered as a fixed starter repo (`doc-agent-starter.zip`). **Students choose models & parameters (in
`configs/`); they do not choose where code goes.** Implement only inside `# IMPLEMENT` blocks; signatures,
contracts, pipeline order, and tool names are locked and CI-enforced (`tests/test_structure.py`).
The repo ships `traces/run.jsonl` (the JSONL trajectory) and `needs_research` task tags in
`grading_kit/tasks.jsonl`, which the A3 agentic gate reads to confirm evidence-gated re-search fires.

**Layout.** Mandatory homes live in `src/doc_agent/` (every group implements): `ingest/` (loader,
preprocess, **enhance** = VAE/diffusion), `vision/` (layout, ocr), `index/`, `retrieval/`, `agent/`
(agent loop, tools, memory; guardrails/pii SHIP as the Secure/Private NFRs; hitl/judge ship as bonus), `llm/` (client, prompts, postprocess),
`rl/` (env, policy, rlvr, train), `training/` (datamodule, lit_modules, train, adapt=LoRA/quantize),
`eval/` (metrics, ablation, robustness, fairness, interpret, judge, calibration), `serve/` (api, ui),
`mlops/` (tracking, registry, monitor), `governance/` (pii), `data/` (versioning, validate), plus
`contracts.py`, `settings.py`, `logging_conf.py`, `pipeline.py`. Optional, **profile-gated** features live
separately in `src/doc_agent/optional/` (synthetic data, stream ingest, query expansion, cache, api
security, retrain trigger, config snapshot, DP) — OFF by default, not required by the structure gate.

**Every deliverable is a pre-named stub.** The repo ships blank stubs for everything a milestone asks students to submit — `notebooks/eda.ipynb`, `notebooks/kb_demo.ipynb`, `scripts/{get_data,build_index,run}.sh`, `data/provenance.md`, `reports/{pipeline_diagram,eval_report,final_report,brochure}.md` + `reports/{demo_link,video_link}.txt` + `reports/figures/`, and `traces/`, `bonus/`, `transcripts/`, plus **`grading_kit/`** (manifest.yaml + heldout_pages/ + labels.jsonl) — the single folder that declares the three axes and makes the project reproducible and gradable; it is also the students' own evaluation harness. Students **edit** these; they never create a new file. The four assignment forms' submission structures reference these exact paths.

Likewise, **every enhancement in the backlog has a fixed home file** in the same stub — students only ever fill a marked body, never add a file. The complete enhancement→home index (E1–E39) lives in the **Codebase Guide, Part 2** (organised per assignment: mandatory → data-speciality → NFR), and mirrors **CLAUDE.md §10**; this specification stays at the stage level and defers the file-by-file map to that single source.

**Enforced best practices.** Pinned lockfile + Docker (reproducible); `.env`/`settings.py` for secrets;
structured logging (auditable); `governance/pii.py` (private); `eval/calibration.py` (calibrated);
`docs/model_card.md` (governance). CI gates: ruff · black · mypy · structure-lock · contracts · tools ·
per-stage unit tests · data tests · smoke · dependency+code security scan. Every stage's code is checked
against an 82-point capability rubric (67 mandatory, 14 optional).

## Part I — How it flows into the four milestones

High-level arc of the four milestones; the assignment forms carry the detail.
- **A1 — Problem & Data:** domain + scanned corpus, licensing/PII, vision/OCR difficulty analysis, eval set,
  agent-on-paper (incl. the retrieval tool + the RL policy it will train).
- **A2 — Build the Knowledge Base:** ingest/enhance/layout/OCR/index built; component choices justified per
  stage; evidence it reads and retrieves (OCR quality + a working retrieval).
- **A3 — Build the Agent & Evaluate:** retrieval+rerank, the agent loop+tools, grounding (guardrails/PII/HITL are conditional, not universal),
  cross-cutting wired; recall@k, grounded-answer quality, the task suite, primary-NFR result, ablations; RL/RLVR bonus.
- **A4 — MLOps & Demo:** ship the document-QA agent; robustness/OOD, PII governance, interpretability; the
  principle scan; the pitch.

**How milestones are submitted.** Each team downloads the starter zip from Moodle, unzips it, and pushes that folder to **one public GitHub repository** of their own, then submits by pushing a tag — `a1-submit`, `a2-submit`, `a3-submit`, `a4-submit`. The commit history *is* the submission; the timeline (and per-member attribution) is read from GitHub's servers, so it can't be back-dated. Each milestone is the previous one plus that milestone's new work. *(2026 cohort: A1 was submitted on Moodle before the stub existed, so the first tag is `a2-submit`, and A1's artifacts are carried into A2.)* See **How-To-Submit** for the exact workflow.


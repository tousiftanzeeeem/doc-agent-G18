# doc-agent — start here

**New here? Read `SUBMISSION.md` first (how to submit), then `handbook/01-START-HERE.pdf`.**

- **Submit via GitHub:** this folder *is* your repo. Create a **public** GitHub repo, `git push`, and each
  milestone `git tag aN-submit && git push --tags`. Full steps: **`SUBMISSION.md`**. A private repo = no submission.
- **Read in order:** `handbook/01-START-HERE` → `02-How-To-Submit` → `03-Project-Specification` →
  `04-Project-Walkthrough` → `05-Codebase-Guide` → `06`-Group-Assignment-Workbook (domains, specialities, NFRs, sources, build buckets).
- **Fill each milestone's form** in `forms/AN_form.docx` and commit it.

---

# doc-agent — regulated starter repo (scanned-document Agentic-RAG)

A fixed skeleton. **You choose models & parameters (in `configs/`). You do NOT choose where code goes.**
Implement only inside functions marked `# IMPLEMENT`. Do not move, rename, or add top-level modules.
CI rejects a repo whose structure or interfaces drift (`tests/test_structure.py`).

## Phase → file map
| Phase | Where |
|---|---|
| 0 Problem/config | `configs/task.yaml`, `configs/config.yaml` |
| 1 Ingestion | `src/doc_agent/ingest/loader.py`, `preprocess.py` |
| 1 Enhancement (VAE/diffusion) | `src/doc_agent/ingest/enhance.py` |
| 2 Layout detection | `src/doc_agent/vision/layout.py` |
| 3 OCR / HTR | `src/doc_agent/vision/ocr.py` |
| 4 Index (chunk/embed/store) | `src/doc_agent/index/` |
| 5 Retrieval | `src/doc_agent/retrieval/` |
| 6 Agent (query→answer) | `src/doc_agent/agent/agent.py`, `tools.py` |
| 6 HITL | `src/doc_agent/agent/hitl.py` |
| 6 Guardrails/security | `src/doc_agent/agent/guardrails.py` |
| 7 RL policy + RLVR | `src/doc_agent/rl/` |
| Training | `src/doc_agent/training/` |
| 8 Serving | `src/doc_agent/serve/`, `Dockerfile` |
| 9 Validation/eval | `src/doc_agent/eval/`, `tests/` |
| MLOps | `src/doc_agent/mlops/` |
| CI/CD | `.github/workflows/` |
| Eval tasks | `grading_kit/tasks.jsonl`, `grading_kit/success_check.py` |
| Pipeline (fixed order) | `src/doc_agent/pipeline.py` |

## Run
```
make setup        # uv sync (pinned lockfile)
make seed         # deterministic seeds
make ingest index # build the KB
make eval         # metrics on tasks.jsonl
make serve        # FastAPI + Gradio
```
See `STRUCTURE.md` for the rules CI enforces.

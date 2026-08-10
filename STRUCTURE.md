# STRUCTURE — the regulation (CI-enforced)

## You MAY change
- `configs/*.yaml` — model checkpoints, hyperparameters, thresholds, the task spec.
- Function **bodies** marked `# IMPLEMENT`.
- `grading_kit/tasks.jsonl` (your eval set) and `grading_kit/success_check.py` verifiers.

## You MUST NOT change
- Directory layout; file names; module paths.
- Function/class **signatures** and the data contracts in `src/doc_agent/contracts.py`.
- The stage order in `src/doc_agent/pipeline.py`.
- The tool names/signatures in `src/doc_agent/agent/tools.py`.

## Gates (CI fails the build if any fail)
1. `ruff` (lint) · `black --check` (format) · `mypy` (types).
2. `tests/test_structure.py` — required modules/callables exist with correct signatures.
3. `tests/test_contracts.py` — data contracts validate.
4. `tests/test_tools.py` — tool interface conformance.
5. `tests/test_smoke.py` — end-to-end tiny run returns a grounded, cited answer.
6. Reproducibility — seeded run reproduces reported metrics within tolerance.

## Per-stage design table (fill in A2)
Every stage states its choice across 8 facets: problem statement, data, model, methods, design,
development, deployment, MLOps. Template: `configs/design_choices.md`.
## Mandatory vs optional (physical separation)
- **Mandatory** homes live in `src/doc_agent/` (every group implements). The structure gate requires them.
- **Optional** homes live in `src/doc_agent/optional/` — profile-gated, OFF by default, NOT required by the
  structure gate. Implement one only if your data speciality calls for it (see optional/README.md).
- **Governance** (`governance/pii.py`) and **data** (`data/versioning.py`, `data/validate.py`) are mandatory.
- Secrets → `.env` (never committed); typed access via `settings.py`. Logging via `logging_conf.py` (no print).
- All LLM/model calls go through `llm/client.py`; all prompts live in `llm/prompts.py`.

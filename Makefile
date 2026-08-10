.PHONY: setup seed ingest index eval serve test lint
setup: ; uv sync --frozen
seed:  ; python scripts/set_seed.py
ingest:; python scripts/run_ingest.py
index: ; python scripts/run_index.py
eval:  ; python scripts/run_eval.py
serve: ; uvicorn doc_agent.serve.api:app --host 0.0.0.0 --port 8000
lint:  ; ruff check . && black --check . && mypy src
test:  ; pytest

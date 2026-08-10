# Agent traces (A3)

Emit **`traces/run.jsonl`** — one JSON object per agent step — so the autograder can read the trajectory
and verify that evidence-gated re-search (the mandatory agentic behaviour) actually fires at runtime.

Schema (one line per step; matches `contracts.TraceStep`):

    {"step":1,"tool":"retrieve","args":{"query":"...","k":10},"obs":{"top_score":0.28,"k":10}}   # weak (< threshold)
    {"step":2,"tool":"retrieve","args":{"query":"...","k":20},"obs":{"top_score":0.31,"k":20}}   # widened, still weak
    {"step":3,"tool":"retrieve","args":{"query":"...","k":40},"obs":{"top_score":0.33,"k":40}}   # k_max, still weak
    {"step":4,"tool":"answer","args":{},"obs":{"abstained":true}}                                # -> abstain

Your tracer (wired at ON_STEP / ON_TOOL_CALL / AFTER_ANSWER in `logging_conf.register`) writes these lines.
Drop at least one full run here. The **A3 agentic gate** runs your agent on the trigger/control
task pairs you tag with `needs_research` in `grading_kit/tasks.jsonl` and asserts the path varies with observations.

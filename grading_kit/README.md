# grading_kit/ — the one folder that makes this project reproducible and gradable.

- **manifest.yaml** — the single entry point: the three axes (domain, data speciality, primary NFR) + pointers to the corpus,
  the held-out slice, and the build/run/eval commands.
- **heldout_pages/** — page-images set aside, never OCR-trained on.
- **labels.jsonl** — ground-truth transcriptions for the held-out pages (the oracle:
  OCR is scored against them, and fresh grading questions are authored from them).

A grader (or you) opens ONLY `manifest.yaml`; it points to everything else. The build/run
scripts and the eval tasks are named there, not copied here, so they never go stale.

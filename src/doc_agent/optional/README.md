# Optional features (profile-gated)
Implement ONLY if your data speciality or NFR requires it. These are NOT part of the mandatory structure gate.
| Module | Needed by profile |
|---|---|
| synthetic_data.py | 11 tiny-label |
| stream_ingest.py | 12 huge-corpus |
| query_expansion.py | 18 recall-first, 27 noisy-query |
| cache.py | 2 low-latency |
| api_security.py | 25 secure, high-stakes |
| retrain_trigger.py | any with drift SLA |
| config_snapshot.py | reproducibility+ |
| dp.py | 22 PII (if model shared) |

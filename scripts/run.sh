#!/usr/bin/env bash
# A3 — reproduce all results end to end
set -euo pipefail
make seed ingest index eval

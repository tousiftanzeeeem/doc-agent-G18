#!/usr/bin/env bash
# A1 — fetch or recreate your scanned corpus into data/raw/
# Source: Internet Archive, item in.ernet.dli.2015.95111
#   "Twenty-second Annual Report Of The Sanitary Commissioner For Bengal(1889)"
#   by W. H. Gregg (1890) — public domain (Digital Library of India collection).
# The 126MB PDF is too large for git, so it lives outside the repo (./corpus/ or here
# in data/raw/, both gitignored) and is fetched/recreated by this script.
set -euo pipefail

CORPUS_PDF="corpus/2015.95111.Twenty-second-Annual-Report-Of-The-Sanitary-Commissioner-For-Bengal1889.pdf"
URL="https://archive.org/download/in.ernet.dli.2015.95111/2015.95111.Twenty-second-Annual-Report-Of-The-Sanitary-Commissioner-For-Bengal1889.pdf"
DEST="data/raw/bengal1889.pdf"

mkdir -p data/raw

if [ -f "$DEST" ]; then
  echo "corpus PDF already present: $DEST"
elif [ -f "$CORPUS_PDF" ]; then
  echo "copying corpus PDF into data/raw/ (gitignored)..."
  cp "$CORPUS_PDF" "$DEST"
else
  echo "downloading corpus PDF from archive.org (126MB)..."
  curl -L --fail --retry 3 -o "$DEST" "$URL"
fi

echo "done. Next: make ingest index   (renders pages, OCRs, embeds, builds the index)"
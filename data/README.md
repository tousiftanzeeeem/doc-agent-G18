# Data contract
- Put scanned page-images in `data/raw/` (gitignored).
- Minimum: **>=300 pages AND >=60,000 words** extracted (huge-corpus profile: >=100k pages).
- Record source URL + license in `configs/task.yaml`. Public-domain / openly licensed only.
- Splits by **document**, not page (leakage rule).

"""Stage 1 — load scanned page-images"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..contracts import Page
from ..logging_conf import get_logger

log = get_logger("ingest.loader")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
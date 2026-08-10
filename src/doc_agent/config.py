"""FIXED config loader."""
from __future__ import annotations
import yaml
from pathlib import Path

def load(path: str | Path = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def load_task(path: str | Path = "configs/task.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

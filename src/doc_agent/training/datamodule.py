"""Training — Lightning datamodule"""
from __future__ import annotations
from ..contracts import *  # noqa

import lightning as L

class DocDataModule(L.LightningDataModule):
    def setup(self, stage: str | None = None) -> None:
        raise NotImplementedError("Training: datamodule.setup (split by document)")


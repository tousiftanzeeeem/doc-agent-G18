"""Training — Lightning modules per trainable component"""
from __future__ import annotations
from ..contracts import *  # noqa

import lightning as L

class LitComponent(L.LightningModule):
    """Wrap enhancer / OCR / retriever training. IMPLEMENT training_step + configure_optimizers."""
    def training_step(self, batch, idx):
        raise NotImplementedError("Training: training_step")
    def configure_optimizers(self):
        raise NotImplementedError("Training: optimizer + LR schedule (from cfg)")


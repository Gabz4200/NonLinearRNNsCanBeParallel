"""Metrics logging callback."""

from __future__ import annotations

import lightning as pl
from lightning.pytorch.callbacks import Callback


class MetricsLogger(Callback):
    """Log metrics at epoch boundaries."""

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pass

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pass

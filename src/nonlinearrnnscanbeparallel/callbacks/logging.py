"""Metrics logging callback."""

from __future__ import annotations

import lightning
from lightning.pytorch.callbacks import Callback


class MetricsLogger(Callback):
    """Log metrics at epoch boundaries."""

    def on_train_epoch_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        pass

    def on_validation_epoch_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        pass

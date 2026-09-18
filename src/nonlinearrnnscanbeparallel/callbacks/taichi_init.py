"""Taichi initialization callback for Lightning."""

from __future__ import annotations

import lightning
from lightning.pytorch.callbacks import Callback

from nonlinearrnnscanbeparallel.kernels.taichi.runtime import ensure_initialized


class TaichiInitCallback(Callback):
    """Initialize Taichi on fit start."""

    def on_fit_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        ensure_initialized()

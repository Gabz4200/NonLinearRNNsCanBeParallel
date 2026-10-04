"""Taichi initialization callback for Lightning."""

from __future__ import annotations

import lightning as pl
from lightning.pytorch.callbacks import Callback

from nonlinearrnnscanbeparallel.kernels.taichi.runtime import ensure_initialized


class TaichiInitCallback(Callback):
    """Initialize Taichi on fit start."""

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        ensure_initialized()

"""Training engine - Trainer factory."""

from __future__ import annotations

from typing import Any

import lightning
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

from nonlinearrnnscanbeparallel.callbacks.taichi_init import TaichiInitCallback


def _model_checkpoint() -> ModelCheckpoint:
    return ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
    )


def _early_stopping() -> EarlyStopping:
    return EarlyStopping(monitor="val/loss", patience=10, mode="min")


CALLBACKS = {
    "model_checkpoint": _model_checkpoint,
    "early_stopping": _early_stopping,
    "taichi_init": TaichiInitCallback,
}


def _iter_callback_flags(callbacks_cfg: object) -> list[tuple[str, bool]]:
    if isinstance(callbacks_cfg, dict):
        return [(str(name), bool(enabled)) for name, enabled in callbacks_cfg.items()]
    if isinstance(callbacks_cfg, (list, tuple)):
        return [(str(name), True) for name in callbacks_cfg]
    return []


def create_trainer(cfg: dict[str, Any]) -> lightning.Trainer:
    """Create a Lightning Trainer from config."""
    callbacks: list[Callback] = []

    for name, enabled in _iter_callback_flags(cfg.get("callbacks", {})):
        if not enabled:
            continue
        factory = CALLBACKS.get(name)
        if factory is not None:
            callbacks.append(factory())

    return lightning.Trainer(
        accelerator=cfg.get("accelerator", "auto"),
        devices=cfg.get("devices", 1),
        max_epochs=cfg.get("max_epochs", 50),
        precision=cfg.get("precision", "32-true"),
        log_every_n_steps=cfg.get("log_every_n_steps", 10),
        gradient_clip_val=cfg.get("gradient_clip_val", 1.0),
        callbacks=callbacks,
        fast_dev_run=cfg.get("fast_dev_run", False),
    )

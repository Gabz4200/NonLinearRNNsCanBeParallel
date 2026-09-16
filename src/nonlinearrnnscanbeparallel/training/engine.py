"""Training engine - Trainer factory."""

from __future__ import annotations

from typing import Any
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from nonlinearrnnscanbeparallel.callbacks.logging import MetricsLogger
from nonlinearrnnscanbeparallel.callbacks.taichi_init import TaichiInitCallback


def create_trainer(cfg: dict[str, Any]) -> L.Trainer:
    """Create a Lightning Trainer from config."""
    callbacks = []
    
    callbacks_cfg = cfg.get("callbacks", {})
    callbacks_items = callbacks_cfg.items() if isinstance(callbacks_cfg, dict) else ((k, True) for k in callbacks_cfg)
    for name, enabled in callbacks_items:
        if not enabled:
            continue
        if name == "model_checkpoint":
            callbacks.append(ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                filename="best-{epoch:02d}-{val_loss:.4f}",
            ))
        elif name == "early_stopping":
            callbacks.append(EarlyStopping(
                monitor="val/loss",
                patience=10,
                mode="min",
            ))
        elif name == "taichi_init":
            callbacks.append(TaichiInitCallback())
        elif name == "metrics_logger":
            callbacks.append(MetricsLogger())
    
    return L.Trainer(
        accelerator=cfg.get("accelerator", "auto"),
        devices=cfg.get("devices", 1),
        max_epochs=cfg.get("max_epochs", 50),
        precision=cfg.get("precision", "32-true"),
        log_every_n_steps=cfg.get("log_every_n_steps", 10),
        gradient_clip_val=cfg.get("gradient_clip_val", 1.0),
        callbacks=callbacks,
    )
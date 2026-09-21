"""Tests for the USTCON Lightning training seam."""

from __future__ import annotations

import lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from nonlinearrnnscanbeparallel.models.parallel_wrapper import get_cosine_schedule_with_warmup
from nonlinearrnnscanbeparallel.tasks.graph_reachability import (
    BPTTGraphReachabilityTask,
    MetricsCallback,
)


class _FixedBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, state=None):
        return self.output(x), None


def test_bptt_validation_reports_aligned_answer_metrics() -> None:
    task = BPTTGraphReachabilityTask(
        {
            "name": "mlp_rnn",
            "vocab_size": 12,
            "hidden_dim": 4,
            "num_layers": 1,
            "num_heads": 1,
            "dropout": 0.0,
            "num_classes": 2,
            "seq_len": 5,
            "max_seq_len": 5,
        },
        _FixedBackbone(),
    )
    samples = [
        {
            "input_ids": torch.tensor([1, 9, 0, 0, 0]),
            "labels": torch.tensor([-100, 0, -100, -100, -100]),
            "reachability_label": torch.tensor(0),
        },
        {
            "input_ids": torch.tensor([0, 0, 0, 9, 0]),
            "labels": torch.tensor([-100, -100, -100, 1, -100]),
            "reachability_label": torch.tensor(1),
        },
    ]
    loader = DataLoader(samples, batch_size=2)
    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[metrics_callback],
    )

    trainer.validate(task, dataloaders=loader, verbose=False)
    metrics = trainer.callback_metrics

    assert metrics["val/reachability_acc"] == metrics["val/accuracy"]
    assert "val/loss" in metrics
    assert len(metrics_callback.train_losses) == 0
    assert len(metrics_callback.val_losses) == 1
    assert len(metrics_callback.val_accuracies) == 1
    assert len(metrics_callback.val_reachability_accs) == 1


def test_metrics_callback_records_fit_epochs_once() -> None:
    task = BPTTGraphReachabilityTask(
        {
            "name": "mlp_rnn",
            "vocab_size": 12,
            "hidden_dim": 4,
            "num_layers": 1,
            "num_heads": 1,
            "dropout": 0.0,
            "num_classes": 2,
            "seq_len": 5,
            "max_seq_len": 5,
        },
        _FixedBackbone(),
    )
    samples = [
        {
            "input_ids": torch.tensor([1, 9, 0, 0, 0]),
            "labels": torch.tensor([-100, 0, -100, -100, -100]),
            "reachability_label": torch.tensor(0),
        },
        {
            "input_ids": torch.tensor([0, 0, 0, 9, 0]),
            "labels": torch.tensor([-100, -100, -100, 1, -100]),
            "reachability_label": torch.tensor(1),
        },
    ]
    loader = DataLoader(samples, batch_size=2)
    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[metrics_callback],
    )

    trainer.fit(task, train_dataloaders=loader, val_dataloaders=loader)

    assert len(metrics_callback.train_losses) == 1
    assert len(metrics_callback.val_losses) == 1
    assert len(metrics_callback.val_accuracies) == 1
    assert len(metrics_callback.val_reachability_accs) == 1


def test_bptt_scheduler_uses_configured_total_steps(monkeypatch) -> None:
    task = BPTTGraphReachabilityTask(
        {
            "name": "mlp_rnn",
            "vocab_size": 12,
            "hidden_dim": 4,
            "num_layers": 1,
            "num_heads": 1,
            "dropout": 0.0,
            "num_classes": 2,
            "seq_len": 5,
            "max_seq_len": 5,
            "total_steps": 7,
        },
        _FixedBackbone(),
    )
    captured: dict[str, int] = {}
    original_scheduler = get_cosine_schedule_with_warmup

    def capture_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio):
        captured.update(
            warmup_steps=num_warmup_steps,
            total_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio,
        )
        return original_scheduler(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio,
        )

    monkeypatch.setattr(
        "nonlinearrnnscanbeparallel.tasks.graph_reachability.get_cosine_schedule_with_warmup",
        capture_scheduler,
    )

    optimizer_config = task.configure_optimizers()

    assert captured == {"warmup_steps": 1, "total_steps": 7, "min_lr_ratio": 0.01}
    assert optimizer_config["lr_scheduler"]["scheduler"] is not None

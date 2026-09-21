"""Lightning task + metric callbacks for the USTCON graph-reachability benchmark.

Core models in `nonlinearrnnscanbeparallel.models` are task-agnostic and only emit
logits + recurrent states. The objective (binary reachability), the BPTT vs.
parallel forward dispatch, the per-group gradient clipping, and the comparison
metric callbacks live here so the unified `scripts/train.py` stays generic.
"""

from __future__ import annotations

import time
from typing import Any

import lightning as pl
import torch
from lightning.pytorch.callbacks import Callback
from torch import nn

from ..data.graph_reachability import ANSWER_MARKER
from ..losses.classification import cross_entropy_loss
from ..models.base import RNNState, RNNStateList
from ..models.parallel_wrapper import get_cosine_schedule_with_warmup


class GradientClippingCallback(Callback):
    """Run wrapper-specific gradient clipping before each optimizer step."""

    def __init__(self, clip_fn) -> None:
        self.clip_fn = clip_fn

    def on_before_optimizer_step(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer
    ) -> None:
        self.clip_fn()


class MetricsCallback(Callback):
    """Collect the metrics serialized into the comparison JSON files."""

    def __init__(self) -> None:
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_accuracies: list[float] = []
        self.val_reachability_accs: list[float] = []
        self.epoch_times: list[float] = []
        self.epoch_start_time: float | None = None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.epoch_start_time is not None:
            self.epoch_times.append(time.time() - self.epoch_start_time)
        metrics = trainer.callback_metrics
        if "train/loss_epoch" in metrics:
            self.train_losses.append(float(metrics["train/loss_epoch"]))

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if "val/loss" in metrics:
            self.val_losses.append(float(metrics["val/loss"]))
        if "val/accuracy" in metrics:
            self.val_accuracies.append(float(metrics["val/accuracy"]))
        if "val/reachability_acc" in metrics:
            self.val_reachability_accs.append(float(metrics["val/reachability_acc"]))


class GraphReachabilityTask(pl.LightningModule):
    """Lightning task for the USTCON graph-reachability benchmark.

    Core model is task-agnostic and injected via ``backbone``; only the head,
    loss mapping, and metric callbacks live here.
    """

    def __init__(self, model_spec: dict[str, Any], backbone: nn.Module) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.input_embed = nn.Embedding(model_spec["vocab_size"], model_spec["hidden_dim"])
        self.criterion = cross_entropy_loss
        # Chunked BPTT boundary; 0 disables chunking (full-sequence BPTT).
        self._max_seq_len = int(model_spec.get("bptt_max_seq_len", model_spec.get("seq_len", 256)))
        self._total_steps = max(1, int(model_spec.get("total_steps", 1)))
        self.model_spec = model_spec

    def forward(self, x, state=None):
        return self.backbone(self.input_embed(x), state)

    def _forward_bptt(
        self, input_ids: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, RNNStateList | None]:
        """Chunked BPTT with state detach between chunks for MLP-RNN stability."""
        _, sequence_len = input_ids.shape
        chunk = self._max_seq_len
        if chunk <= 0 or sequence_len <= chunk:
            return self.forward(input_ids, state)

        outputs: list[torch.Tensor] = []
        for start in range(0, sequence_len, chunk):
            end = min(start + chunk, sequence_len)
            logits, state = self.forward(input_ids[:, start:end], state)
            outputs.append(logits)
            if state is not None and end < sequence_len:
                detached = [
                    RNNState(
                        hidden=item.hidden.detach(),
                        extra=(
                            {k: v.detach() for k, v in item.extra.items()}
                            if item.extra is not None
                            else None
                        ),
                    )
                    for item in state.states
                ]
                state = RNNStateList(detached)
        return torch.cat(outputs, dim=1), state

    def _forward(
        self, input_ids: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, Any]:
        """Full-sequence forward; parallel wrapper handles chunking."""
        return self.forward(input_ids, state)

    def training_step(  # pyright: ignore [reportIncompatibleMethodOverride]
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        logits, _ = self._forward(batch["input_ids"])
        loss = self.criterion(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _answer_metrics(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        reachability_labels: torch.Tensor,
    ) -> torch.Tensor:
        answer_counts = (input_ids == ANSWER_MARKER).sum(dim=1)
        if not torch.all(answer_counts == 1):
            raise ValueError("each USTCON sample must contain exactly one ANSWER token")

        answer_positions = (input_ids == ANSWER_MARKER).nonzero(as_tuple=False)
        row_indices = answer_positions[:, 0]
        sequence_indices = answer_positions[:, 1]
        predictions = logits[row_indices, sequence_indices].argmax(dim=-1)
        return (predictions == reachability_labels[row_indices]).float().mean()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        input_ids = batch["input_ids"]
        logits, _ = self._forward(input_ids)
        loss = self.criterion(logits, batch["labels"])
        reachability_acc = self._answer_metrics(input_ids, logits, batch["reachability_label"])
        self.log("val/reachability_acc", reachability_acc, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", reachability_acc, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self):  # pyright: ignore [reportIncompatibleMethodOverride]
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        weight_decay = float(self.model_spec.get("weight_decay", 0.01))
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=float(self.model_spec.get("lr", 1e-3)),
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        total_steps = max(1, int(getattr(self, "_total_steps", 1)))
        warmup_steps = max(1, int(total_steps * float(self.model_spec.get("warmup_ratio", 0.1))))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=float(self.model_spec.get("min_lr_ratio", 0.01)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class ParallelGraphReachabilityTask(GraphReachabilityTask):
    """Parallel chunkwise training forward dispatch."""

    def _forward(self, input_ids: torch.Tensor, state=None):
        return self.forward(input_ids, state)


class BPTTGraphReachabilityTask(GraphReachabilityTask):
    """Standard full-/chunked-sequence BPTT baseline."""

    def _forward(self, input_ids: torch.Tensor, state=None):
        return self._forward_bptt(input_ids, state)

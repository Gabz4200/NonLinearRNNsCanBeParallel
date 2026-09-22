"""Lightning module with BPTT for RNN training.

Separates backbone (model) construction from task objective.
The model is injected via `model_spec` (name + kwargs) or a
pre-built backbone, avoiding model-specific kwarg filtering.
"""

from __future__ import annotations

from typing import Any

import lightning
import torch
from torch import nn

from ..models.base import RNNModule, RNNStateList, chunked_forward_bptt
from ..models.registry import get_model


class RNNTask(lightning.LightningModule):
    """Lightning task for any model implementing RNNModule protocol."""

    def __init__(
        self,
        model_spec: dict[str, Any] | None = None,
        backbone: RNNModule | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        model_spec = model_spec or {}
        model_name = model_spec.get("name", "mlp_rnn")
        vocab_size = model_spec.get("vocab_size", 36)
        hidden_dim = model_spec.get("hidden_dim", 64)
        max_seq_len = model_spec.get("max_seq_len", 66)

        if backbone is not None:
            self.backbone: nn.Module = backbone  # type: ignore[assignment]
        else:
            model_kwargs = {
                k: v
                for k, v in model_spec.items()
                if k not in ("name", "vocab_size", "hidden_dim", "max_seq_len")
            }
            model_kwargs["input_dim"] = hidden_dim
            model_kwargs["hidden_dim"] = hidden_dim
            self.backbone = get_model(model_name, **model_kwargs)

        self.input_embed = nn.Embedding(vocab_size, hidden_dim)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.bptt_max_seq_len = max_seq_len

    def _forward_chunked(
        self, input_ids: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, RNNStateList | None]:
        return chunked_forward_bptt(self.forward, input_ids, state, self.bptt_max_seq_len)

    def forward(
        self, x: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, RNNStateList]:
        x = self.input_embed(x)
        return self.backbone(x, state)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        logits, _ = self._forward_chunked(input_ids)
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        logits, _ = self._forward_chunked(input_ids)
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        eos_mask = labels != -100
        if eos_mask.any():
            eos_idx = eos_mask.int().argmax(dim=1)
            preds = logits[torch.arange(logits.size(0)), eos_idx, :].argmax(dim=-1)
            acc_mask = eos_mask.sum(dim=1) > 0
            if acc_mask.any():
                pred_eos = preds[acc_mask]
                label_eos = labels[torch.arange(labels.size(0)), eos_idx][acc_mask]
                acc = (pred_eos == label_eos).float().mean()
                self.log("val/accuracy", acc, on_epoch=True, prog_bar=True)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> Any:
        decay = []
        no_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name == "input_embed.weight":
                no_decay.append(param)
            elif param.dim() == 2:
                decay.append(param)
            else:
                no_decay.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 0.01},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

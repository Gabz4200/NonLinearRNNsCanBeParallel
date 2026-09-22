"""Lightning task for causal language modeling (BPTT and parallel modes).

Core models in `nonlinearrnnscanbeparallel.models` are task-agnostic. The
LM objective (shifted cross-entropy), the BPTT vs. parallel forward
dispatch, and perplexity/accuracy logging live here.
"""

from __future__ import annotations

from typing import Any

import lightning as pl
import torch
from torch import nn

from ..logging.metrics import perplexity, token_accuracy
from ..losses.language_modeling import causal_lm_loss
from ..models.base import RNNStateList, chunked_forward_bptt
from ..models.parallel_wrapper import get_cosine_schedule_with_warmup


class LMLightningTask(pl.LightningModule):
    """Causal LM task over any backbone.

    Backbones without an internal embedding (legacy RNNs) get
    ``input_embed``; :class:`NanoRNN` owns ``wte`` so token IDs feed
    straight through (``input_embed`` is None).
    """

    def __init__(self, model_spec: dict[str, Any], backbone: nn.Module) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        # Tokens feed straight through when the backbone (or the RNN it
        # wraps) owns an embedding; only legacy RNNs get ``input_embed``.
        wrapped = getattr(backbone, "target", None)
        owns_embedding = hasattr(backbone, "wte") or (
            wrapped is not None and hasattr(wrapped, "wte")
        )
        self.input_embed = (
            None
            if owns_embedding
            else nn.Embedding(model_spec["vocab_size"], model_spec["hidden_dim"])
        )
        self.criterion = causal_lm_loss
        self._max_seq_len = int(
            model_spec.get(
                "bptt_max_seq_len",
                model_spec.get("block_size", model_spec.get("seq_len", 1024)),
            )
        )
        self._total_steps = max(1, int(model_spec.get("total_steps", 1)))
        self.model_spec = model_spec

    def forward(
        self, x: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, Any]:
        if self.input_embed is not None:
            return self.backbone(self.input_embed(x), state)
        return self.backbone(x, state)

    def _forward_bptt(
        self, input_ids: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, RNNStateList | None]:
        """Chunked BPTT with state detach between chunks."""
        return chunked_forward_bptt(self.forward, input_ids, state, self._max_seq_len)

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

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        logits, _ = self._forward(batch["input_ids"])
        loss = self.criterion(logits, batch["labels"])
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/ppl", perplexity(float(loss)), on_epoch=True, prog_bar=True)
        self.log(
            "val/acc",
            token_accuracy(logits[:, :-1], batch["labels"][:, 1:]),
            on_epoch=True,
            prog_bar=True,
        )

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
            lr=float(self.model_spec.get("lr", 3e-4)),
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        total_steps = max(1, int(getattr(self, "_total_steps", 1)))
        warmup_steps = max(1, int(total_steps * float(self.model_spec.get("warmup_ratio", 0.05))))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=float(self.model_spec.get("min_lr_ratio", 0.1)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class BPTTLMTask(LMLightningTask):
    """Standard full-/chunked-sequence BPTT baseline."""

    def _forward(
        self, input_ids: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, Any]:
        return self._forward_bptt(input_ids, state)

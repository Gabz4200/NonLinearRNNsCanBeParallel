"""HuggingFace PreTrainedModel wrapper for NonLinearRNNsCanBeParallel."""

from __future__ import annotations

import torch
from torch import nn
from transformers import BaseModelOutput, PreTrainedModel

from nonlinearrnnscanbeparallel.models.registry import get_model

from .configuration_nonlinearrnnscanbeparallel import NonLinearRNNsCanBeParallelConfig


class NonLinearRNNsCanBeParallelModel(PreTrainedModel):
    """HF wrapper around native MLP-RNN / rKAN-RNN."""

    config_class = NonLinearRNNsCanBeParallelConfig
    base_model_prefix = "nonlinearrnnscanbeparallel"

    def __init__(self, config: NonLinearRNNsCanBeParallelConfig):
        super().__init__(config)
        # Wraps the native model. Does not duplicate weights or code.
        self.native = get_model(
            config.model_name,
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> BaseModelOutput | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden = inputs_embeds if inputs_embeds is not None else input_ids
        if hidden is None:
            raise ValueError("Specify input_ids or inputs_embeds")
        out = self.native(hidden)
        if return_dict:
            return BaseModelOutput(last_hidden_state=out[0])
        return out

    def get_input_embeddings(self) -> nn.Module | None:
        # Native model has no embedding; identity by convention.
        return None

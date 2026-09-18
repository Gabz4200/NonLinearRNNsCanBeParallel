"""HuggingFace PretrainedConfig for NonLinearRNNsCanBeParallel."""

from __future__ import annotations

from transformers import PretrainedConfig


class NonLinearRNNsCanBeParallelConfig(PretrainedConfig):
    """Configuration for MLP-RNN / rKAN-RNN models."""

    model_type = "nonlinearrnnscanbeparallel"
    is_composition = False

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        model_name: str = "mlp_rnn",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.model_name = model_name

"""Auto-registration for HuggingFace Transformers."""

from __future__ import annotations

from transformers import AutoConfig, AutoModel

from .configuration_nonlinearrnnscanbeparallel import NonLinearRNNsCanBeParallelConfig
from .modeling_nonlinearrnnscanbeparallel import NonLinearRNNsCanBeParallelModel


def register_auto_classes() -> None:
    """Register config and model classes with AutoConfig/AutoModel."""
    AutoConfig.register("nonlinearrnnscanbeparallel", NonLinearRNNsCanBeParallelConfig)
    AutoModel.register(NonLinearRNNsCanBeParallelConfig, NonLinearRNNsCanBeParallelModel)


# Register on import so Auto classes resolve.
register_auto_classes()

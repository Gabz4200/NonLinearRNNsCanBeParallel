"""HuggingFace Transformers integration - triggers auto-registration."""

from .auto_mapping import register_auto_classes
from .configuration_nonlinearrnnscanbeparallel import NonLinearRNNsCanBeParallelConfig
from .modeling_nonlinearrnnscanbeparallel import NonLinearRNNsCanBeParallelModel

__all__ = [
    "NonLinearRNNsCanBeParallelConfig",
    "NonLinearRNNsCanBeParallelModel",
    "register_auto_classes",
]

register_auto_classes()

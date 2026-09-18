"""Model registry for config-driven architecture selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a model constructor under a name."""

    def decorator(cls: Callable[..., Any]) -> Callable[..., Any]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **kwargs: Any) -> Any:
    """Get a registered model by name and instantiate it."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not registered. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """List all registered model names."""
    return list(MODEL_REGISTRY.keys())

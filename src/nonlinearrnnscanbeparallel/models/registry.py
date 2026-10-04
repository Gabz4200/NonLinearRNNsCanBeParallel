"""Model registry for config-driven architecture selection."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}

# Populating the registry is a side effect of importing the models package, and
# importing it here would be circular: the model modules import register_model
# from this file. So the first lookup triggers the import instead, which means
# callers need only import this module -- no separate "import for side effects"
# line at every entry point.
_LOADED = False


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    importlib.import_module("nonlinearrnnscanbeparallel.models")


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
    _ensure_loaded()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not registered. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """List all registered model names."""
    _ensure_loaded()
    return list(MODEL_REGISTRY.keys())

"""Shared helpers for the Hydra-driven scripts."""

from __future__ import annotations

from typing import Any, cast

from omegaconf import OmegaConf

# Training-only model keys forwarded to the Lightning task, not to get_model().
TRAINING_KEYS = {
    "lr",
    "weight_decay",
    "warmup_ratio",
    "min_lr_ratio",
    "total_steps",
    "target_grad_clip",
    "scaffold_grad_clip",
    "translator_grad_clip",
    "bptt_max_seq_len",
}


def node_dict(node: Any) -> dict[str, Any]:
    out = OmegaConf.to_container(node, resolve=True)
    assert isinstance(out, dict)
    return cast(dict[str, Any], dict(out))

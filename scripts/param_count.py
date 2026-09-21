"""Print parameter counts for a model config and assert Tiny budget.

Usage:
    python scripts/param_count.py --config-name config_lm model=nano_min_gru_tiny
    python scripts/param_count.py --config-name config model=nano_min_gru_tiny
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from nonlinearrnnscanbeparallel.models.parallel_wrapper import ParallelRNNTrainer
from nonlinearrnnscanbeparallel.models.registry import get_model

_TRAINING_KEYS = {
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

_MIN_TARGET = 124_000_000
_MAX_TARGET = 350_000_000


def _dict(node: Any) -> dict[str, Any]:
    out = OmegaConf.to_container(node, resolve=True)
    assert isinstance(out, dict)
    return cast(dict[str, Any], dict(out))


def _count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@hydra.main(version_base=None, config_path="../configs", config_name="config_lm")
def main(cfg: DictConfig) -> None:
    model_cfg = _dict(cfg.get("model", {}))
    parallel_cfg = _dict(cfg.get("parallel", {}))
    name = model_cfg.get("name", "nano_rnn")

    model_kwargs = {k: v for k, v in model_cfg.items() if k not in ("name", *_TRAINING_KEYS)}
    model_kwargs.setdefault("input_dim", int(model_cfg.get("hidden_dim", 768)))
    target = get_model(name, **model_kwargs)

    def cell_type() -> str:
        if name == "nano_rnn":
            return str(model_cfg.get("mixer_type", "min_gru"))
        return {
            "mlp_rnn": "mlp",
            "rkan_rnn": "rkan",
            "m2rnn": "m2rnn",
            "min_gru": "min_gru",
            "min_lstm": "min_lstm",
        }.get(name, name)

    chunk = int(parallel_cfg.get("chunk_size", 1024))
    scaffold_dim = int(parallel_cfg.get("scaffold_dim", 256))
    wrapper = ParallelRNNTrainer(
        target_rnn=target,
        chunk_size=chunk,
        scaffold_dim=scaffold_dim,
        target_type=cell_type(),
        scaffold_type=str(parallel_cfg.get("scaffold_type", "min_gru")),
        scaffold_num_layers=int(parallel_cfg.get("scaffold_num_layers", 1)),
        translator_type=(
            str(parallel_cfg["translator_type"])
            if parallel_cfg.get("translator_type") is not None
            else None
        ),
        translator_num_layers=int(parallel_cfg.get("translator_num_layers", 1)),
    )

    target_params = _count(target)
    scaffold_params = _count(wrapper.scaffolds)
    translator_params = _count(wrapper.translators)

    print(f"model                : {name} (mixer={target.mixer_type})")
    print(f"target (sans scaffold): {target_params:>12,} params")
    print(f"scaffold             : {scaffold_params:>12,} params")
    print(f"translator           : {translator_params:>12,} params")
    print(
        f"parallel total       : {target_params + scaffold_params + translator_params:>12,} params"
    )

    if _MIN_TARGET <= target_params <= _MAX_TARGET:
        print(f"OK: target params in [{_MIN_TARGET / 1e6:.0f}M, {_MAX_TARGET / 1e6:.0f}M]")
    else:
        print(
            f"FAIL: target params {target_params / 1e6:.1f}M outside "
            f"[{_MIN_TARGET / 1e6:.0f}M, {_MAX_TARGET / 1e6:.0f}M]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

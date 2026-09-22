"""Unified training entry point: task x models x modes via Hydra.

Replaces train_all_models.py, train_long_sequence.py, train_ustcon*.py.
Choose the task, the models in the run, and whether to use parallel, BPTT,
or both -- all from the command line or the config files.

Usage:
  # default config (sorted_graph_connectivity, mlp_rnn, parallel)
  python scripts/train.py

  # USTCON comparison: all models, parallel + BPTT
  python scripts/train.py --config-name config_ustcon

  # compare every model on long_sequence, parallel only
  python scripts/train.py --config-name config_long_sequence modes=[parallel]

  # single model / single mode
  python scripts/train.py --config-name config_ustcon models=[rkan_rnn] modes=[bptt]

  # scale down for a quick run; adjust any hyperparam by dotted overrides
  python scripts/train.py --config-name config_ustcon \
    models=[mlp_rnn] modes=[parallel] \
    data.num_samples=64 data.max_seq_len=128 \
    model.hidden_dim=32 model.num_layers=1 \
    trainer.max_epochs=1 trainer.accumulate_grad_batches=1 \
    fast_dev_run=false
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hydra
import lightning as pl
import torch
from _util import TRAINING_KEYS, node_dict
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf

from nonlinearrnnscanbeparallel.data.datamodule import GraphConnectivityDataModule
from nonlinearrnnscanbeparallel.data.graph_reachability import (
    GraphReachabilityConfig,
    GraphReachabilityDataModule,
)
from nonlinearrnnscanbeparallel.data.long_sequence import LongSequenceLMDataModule
from nonlinearrnnscanbeparallel.data.openthoughts_lm import OpenThoughtsLMDataModule
from nonlinearrnnscanbeparallel.models.nano_rnn import NanoRNN
from nonlinearrnnscanbeparallel.models.parallel_wrapper import (
    ParallelRNNTrainer,
    configure_gradient_clipping,
    get_cosine_schedule_with_warmup,
)
from nonlinearrnnscanbeparallel.models.registry import get_model
from nonlinearrnnscanbeparallel.modules.lightning_module import RNNTask
from nonlinearrnnscanbeparallel.tasks.graph_reachability import (
    BPTTGraphReachabilityTask,
    GradientClippingCallback,
    GraphReachabilityTask,
    MetricsCallback,
)
from nonlinearrnnscanbeparallel.tasks.language_modeling import BPTTLMTask, LMLightningTask
from nonlinearrnnscanbeparallel.training.engine import create_trainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_CELL_TYPE = {
    "mlp_rnn": "mlp",
    "rkan_rnn": "rkan",
    "m2rnn": "m2rnn",
    "min_gru": "min_gru",
    "min_lstm": "min_lstm",
}


def _cell_type(model_name: str, model_cfg: dict[str, Any]) -> str:
    """Resolve the wrapper ``target_type`` (cell) from the model config.

    nano_rnn selects its mixer explicitly; legacy registry names map to
    their cell type via ``_CELL_TYPE``.
    """
    if model_name == "nano_rnn":
        return str(model_cfg.get("mixer_type", "min_gru"))
    explicit = model_cfg.get("target_type")
    if explicit is not None:
        return str(explicit)
    return _CELL_TYPE.get(model_name, model_name)


def _build_datamodule(task: str, data_cfg: dict[str, Any]) -> pl.LightningDataModule:
    if task == "ustcon":
        data_cfg.setdefault("max_seq_len", 256)
        return GraphReachabilityDataModule(GraphReachabilityConfig(**data_cfg))
    if task == "long_sequence":
        return LongSequenceLMDataModule(data_cfg)
    if task == "openthoughts_lm":
        data_cfg.pop("task", None)
        return OpenThoughtsLMDataModule(data_cfg)
    return GraphConnectivityDataModule(data_cfg)


def _rnn_task_optimizers(lit_task: pl.LightningModule, model_cfg: dict, total_steps: int) -> None:
    """Attach config-driven optimizer + warmup/cosine scheduler to a bare RNNTask."""

    def configure_optimizers() -> dict[str, Any]:
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for name, param in lit_task.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": float(model_cfg.get("weight_decay", 0.01))},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=float(model_cfg.get("lr", 1e-3)),
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        warmup_steps = max(1, int(total_steps * float(model_cfg.get("warmup_ratio", 0.1))))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=float(model_cfg.get("min_lr_ratio", 0.01)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    lit_task.configure_optimizers = configure_optimizers  # type: ignore[method-assign]


def _build_task(
    task: str,
    mode: str,
    model_name: str,
    model_cfg: dict,
    parallel_cfg: dict,
    total_steps: int,
    target: torch.nn.Module,
) -> tuple[pl.LightningModule, Callback | None, float]:
    target_type = _cell_type(model_name, model_cfg)
    spec = {**model_cfg, "name": model_name, "total_steps": total_steps}

    def _parallel_backbone(default_chunk: int, default_scaffold: int) -> ParallelRNNTrainer:
        return ParallelRNNTrainer(
            target_rnn=target,
            chunk_size=int(parallel_cfg.get("chunk_size", default_chunk)),
            scaffold_dim=int(parallel_cfg.get("scaffold_dim", default_scaffold)),
            target_type=target_type,
            detach_boundary=bool(parallel_cfg.get("detach_boundary", False)),
            use_gdn2_init=bool(parallel_cfg.get("use_gdn2_init", True)),
            scaffold_type=str(parallel_cfg.get("scaffold_type", "min_gru")),
            scaffold_num_layers=int(parallel_cfg.get("scaffold_num_layers", 1)),
            translator_type=(
                str(parallel_cfg["translator_type"])
                if parallel_cfg.get("translator_type") is not None
                else None
            ),
            translator_num_layers=int(parallel_cfg.get("translator_num_layers", 1)),
        )

    def _clip_callback(
        backbone: ParallelRNNTrainer,
    ) -> GradientClippingCallback:
        return GradientClippingCallback(
            configure_gradient_clipping(
                backbone,
                target_grad_clip=float(model_cfg.get("target_grad_clip", 1.0)),
                scaffold_grad_clip=float(model_cfg.get("scaffold_grad_clip", 0.5)),
                translator_grad_clip=float(model_cfg.get("translator_grad_clip", 0.5)),
            )
        )

    if task == "ustcon":
        if mode == "parallel":
            backbone = _parallel_backbone(64, 128)
            return (
                GraphReachabilityTask(spec, backbone),
                _clip_callback(backbone),
                0.0,
            )
        return (
            BPTTGraphReachabilityTask(spec, target),
            None,
            float(model_cfg.get("target_grad_clip", 1.0)),
        )

    if task == "openthoughts_lm":
        if mode == "parallel":
            backbone = _parallel_backbone(1024, 256)
            return (
                LMLightningTask(spec, backbone),
                _clip_callback(backbone),
                0.0,
            )
        return (
            BPTTLMTask(spec, target),
            None,
            float(model_cfg.get("target_grad_clip", 1.0)),
        )

    backbone: Any = target
    clip_cb: Callback | None = None
    clip_val = float(model_cfg.get("target_grad_clip", 1.0))
    if mode == "parallel":
        backbone = _parallel_backbone(128, 256)
        clip_cb = _clip_callback(backbone)
        clip_val = 0.0
    rnn_task = RNNTask(
        model_spec={
            "name": model_name,
            "vocab_size": model_cfg.get("vocab_size", 64),
            "hidden_dim": model_cfg.get("hidden_dim", 256),
            "num_layers": model_cfg.get("num_layers", 2),
            "num_heads": model_cfg.get("num_heads", 4),
            "dropout": model_cfg.get("dropout", 0.1),
            "num_classes": model_cfg.get("num_classes", 2),
            "max_seq_len": model_cfg.get("max_seq_len", model_cfg.get("seq_len", 256)),
        },
        backbone=backbone,
    )
    _rnn_task_optimizers(rnn_task, model_cfg, total_steps)
    return rnn_task, clip_cb, clip_val


def _run_one(
    task_name: str,
    mode: str,
    model_name: str,
    cfg: DictConfig,
    results_dir: Path,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    data_cfg = node_dict(cfg.get("data", {}))
    model_cfg = node_dict(cfg.get("model", {}))
    parallel_cfg = node_dict(cfg.get("parallel", {}))
    trainer_cfg = node_dict(cfg.get("trainer", {}))
    trainer_cfg["callbacks"] = node_dict(cfg.get("callbacks", {}))
    trainer_cfg["fast_dev_run"] = bool(cfg.get("fast_dev_run", False))

    datamodule = _build_datamodule(task_name, data_cfg)
    datamodule.setup("fit")
    total_steps = max(
        1,
        math.ceil(
            len(datamodule.train_dataloader())
            / max(1, int(trainer_cfg.get("accumulate_grad_batches", 1)))
        )
        * int(trainer_cfg.get("max_epochs", 10)),
    )

    model_kwargs = {k: v for k, v in model_cfg.items() if k not in ("name", *TRAINING_KEYS)}
    model_kwargs.setdefault("input_dim", int(model_cfg.get("hidden_dim", 256)))
    target = get_model(model_name, **model_kwargs)
    if isinstance(target, NanoRNN) and hasattr(datamodule, "tokenizer_vocab_size"):
        target_vocab = int(datamodule.tokenizer_vocab_size)
        if target.vocab_size != target_vocab:
            print(
                f"WARNING: vocab size mismatch (model={target.vocab_size}, "
                f"tokenizer={target_vocab}); resizing tied embeddings/lm_head"
            )
            target.resize_vocab(target_vocab)
    lit_task, clip_cb, clip_val = _build_task(
        task_name, mode, model_name, model_cfg, parallel_cfg, total_steps, target
    )

    trainer_cfg["gradient_clip_val"] = clip_val
    trainer = create_trainer(trainer_cfg)
    metric_cb = MetricsCallback()
    extra = [cb for cb in (clip_cb, metric_cb) if cb is not None]
    if extra:
        cast(Any, trainer).callbacks.extend(extra)

    start = time.time()
    trainer.fit(lit_task, datamodule=datamodule)
    elapsed = time.time() - start

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    result = {
        "model": model_name,
        "mode": mode,
        "time_seconds": elapsed,
        "trainable_params": sum(p.numel() for p in lit_task.parameters() if p.requires_grad),
        "metrics": metrics,
    }
    curves = {
        "train_losses": metric_cb.train_losses,
        "val_losses": metric_cb.val_losses,
    }
    return result, curves


def _style_log_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)


def _plot_results(runs: list[dict[str, Any]], results_dir: Path) -> None:
    """Save per-run loss curves and a validation-loss comparison into results_dir."""
    labeled = [(f"{r['model']} [{r['mode']}]", r["train_losses"], r["val_losses"]) for r in runs]
    if not labeled:
        return
    results_dir.mkdir(parents=True, exist_ok=True)

    n = len(labeled)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    for ax, (label, train, val) in zip(axes[0], labeled, strict=True):
        if train:
            ax.plot(range(len(train)), train, label="train", marker="o")
        if val:
            ax.plot(range(len(val)), val, label="val", marker="s")
        _style_log_axes(ax, "epoch", "loss", label)
    fig.tight_layout()
    fig.savefig(results_dir / "loss_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, train, val in labeled:
        if val:
            ax.plot(range(len(val)), val, label=label, marker="s")
        elif train:
            ax.plot(range(len(train)), train, label=f"{label} (train)", marker="o")
    _style_log_axes(ax, "epoch", "validation loss", "Validation loss comparison")
    fig.tight_layout()
    fig.savefig(results_dir / "val_loss_comparison.png")
    plt.close(fig)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    task_name: str = cfg.get("task", "sorted_graph_connectivity")
    models: list[str] = [str(m) for m in cfg.get("models", ["mlp_rnn"])]
    modes: list[str] = [str(m) for m in (cfg.get("modes") if cfg.get("modes") else ["parallel"])]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = PROJECT_ROOT / "results" / f"{task_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{task_name} x {','.join(models)} x {','.join(modes)}"
    print(f"\n{'=' * 60}\nRUN: {run_id}\nResults: {results_dir}\n{'=' * 60}")

    all_results: list[dict[str, Any]] = []
    curves_by_run: list[dict[str, Any]] = []
    for mode in modes:
        for model_name in models:
            print(f"\n>>> training {model_name} [{mode.upper()}] on {task_name}")
            try:
                result, curves = _run_one(task_name, mode, model_name, cfg, results_dir)
                all_results.append(result)
                curves_by_run.append({"model": model_name, "mode": mode, **curves})
            except Exception as error:  # noqa: BLE001
                print(f"ERROR training {model_name} [{mode}]: {error}")
                traceback.print_exc()
                all_results.append({"model": model_name, "mode": mode, "error": str(error)})

    _plot_results(curves_by_run, results_dir)
    summary = {
        "timestamp": timestamp,
        "task": task_name,
        "models": models,
        "modes": modes,
        "results": all_results,
    }
    with (results_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for result in all_results:
        if "error" in result:
            print(f"{result['model']:10s} [{result['mode']}]: FAILED - {result['error']}")
            continue
        metrics = result["metrics"]
        row = f"{result['model']:10s} [{result['mode']}]: "
        for key in ("val/loss", "val/accuracy", "val/reachability_acc", "train/loss_epoch"):
            if key in metrics:
                row += f"{key}={metrics[key]:.4f} "
            else:
                row += f"{key}=n/a "
        print(f"{row}time={result['time_seconds']:.1f}s")
    print(f"\nFull summary saved to: {results_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

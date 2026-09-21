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

import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf

from nonlinearrnnscanbeparallel.data.datamodule import GraphConnectivityDataModule
from nonlinearrnnscanbeparallel.data.graph_reachability import (
    GraphReachabilityConfig,
    GraphReachabilityDataModule,
)
from nonlinearrnnscanbeparallel.data.long_sequence import LongSequenceLMDataModule
from nonlinearrnnscanbeparallel.models import M2RNN, MLPRNN, RKANRNN  # noqa: F401
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
    MetricsCallback,
    ParallelGraphReachabilityTask,
)
from nonlinearrnnscanbeparallel.training.engine import create_trainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# model name -> wrapper `target_type` (translator/scaffold state update kind)
TARGET_TYPE = {
    "mlp_rnn": "mlp",
    "rkan_rnn": "rkan",
    "m2rnn": "m2rnn",
    "min_gru": "min_gru",
    "min_lstm": "min_lstm",
}


def _dict(node: Any) -> dict[str, Any]:
    out = OmegaConf.to_container(node, resolve=True)
    assert isinstance(out, dict)
    return cast(dict[str, Any], dict(out))


def _build_datamodule(task: str, data_cfg: dict[str, Any]) -> pl.LightningDataModule:
    if task == "ustcon":
        data_cfg.setdefault("max_seq_len", 256)
        return GraphReachabilityDataModule(GraphReachabilityConfig(**data_cfg))
    if task == "long_sequence":
        return LongSequenceLMDataModule(data_cfg)
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
    target_type = TARGET_TYPE.get(model_name, model_name)
    spec = {**model_cfg, "name": model_name, "total_steps": total_steps}

    if task == "ustcon":
        if mode == "parallel":
            backbone = ParallelRNNTrainer(
                target_rnn=target,
                chunk_size=int(parallel_cfg.get("chunk_size", 64)),
                scaffold_dim=int(parallel_cfg.get("scaffold_dim", 128)),
                target_type=target_type,
                detach_boundary=bool(parallel_cfg.get("detach_boundary", False)),
                use_gdn2_init=bool(parallel_cfg.get("use_gdn2_init", True)),
            )
            clip_fn = configure_gradient_clipping(
                backbone,
                target_grad_clip=float(model_cfg.get("target_grad_clip", 1.0)),
                scaffold_grad_clip=float(model_cfg.get("scaffold_grad_clip", 0.5)),
                translator_grad_clip=float(model_cfg.get("translator_grad_clip", 0.5)),
            )
            return (
                ParallelGraphReachabilityTask(spec, backbone),
                GradientClippingCallback(clip_fn),
                0.0,
            )
        return (
            BPTTGraphReachabilityTask(spec, target),
            None,
            float(model_cfg.get("target_grad_clip", 1.0)),
        )

    backbone: Any = target
    clip_cb: Callback | None = None
    clip_val = float(model_cfg.get("target_grad_clip", 1.0))
    if mode == "parallel":
        backbone = ParallelRNNTrainer(
            target_rnn=target,
            chunk_size=int(parallel_cfg.get("chunk_size", 128)),
            scaffold_dim=int(parallel_cfg.get("scaffold_dim", 256)),
            target_type=target_type,
            detach_boundary=bool(parallel_cfg.get("detach_boundary", False)),
            use_gdn2_init=bool(parallel_cfg.get("use_gdn2_init", True)),
        )
        clip_cb = GradientClippingCallback(
            configure_gradient_clipping(
                backbone,
                target_grad_clip=float(model_cfg.get("target_grad_clip", 1.0)),
                scaffold_grad_clip=float(model_cfg.get("scaffold_grad_clip", 0.5)),
                translator_grad_clip=float(model_cfg.get("translator_grad_clip", 0.5)),
            )
        )
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
) -> dict[str, Any]:
    data_cfg = _dict(cfg.get("data", {}))
    model_cfg = _dict(cfg.get("model", {}))
    parallel_cfg = _dict(cfg.get("parallel", {}))
    trainer_cfg = _dict(cfg.get("trainer", {}))
    trainer_cfg["callbacks"] = _dict(cfg.get("callbacks", {}))
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

    target = get_model(
        model_name,
        input_dim=int(model_cfg.get("hidden_dim", 256)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        vocab_size=int(model_cfg.get("vocab_size", 64)),
    )
    lit_task, clip_cb, clip_val = _build_task(
        task_name, mode, model_name, model_cfg, parallel_cfg, total_steps, target
    )

    trainer_cfg["gradient_clip_val"] = clip_val
    trainer = create_trainer(trainer_cfg)
    extra: list[Callback] = [clip_cb] if clip_cb is not None else []
    if task_name == "ustcon":
        extra.append(MetricsCallback())
    if extra:
        cast(Any, trainer).callbacks.extend(extra)

    start = time.time()
    trainer.fit(lit_task, datamodule=datamodule)
    elapsed = time.time() - start

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    return {
        "model": model_name,
        "mode": mode,
        "time_seconds": elapsed,
        "trainable_params": sum(p.numel() for p in lit_task.parameters() if p.requires_grad),
        "metrics": metrics,
    }


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
    for mode in modes:
        for model_name in models:
            print(f"\n>>> training {model_name} [{mode.upper()}] on {task_name}")
            try:
                all_results.append(_run_one(task_name, mode, model_name, cfg, results_dir))
            except Exception as error:  # noqa: BLE001
                import traceback

                print(f"ERROR training {model_name} [{mode}]: {error}")
                traceback.print_exc()
                all_results.append({"model": model_name, "mode": mode, "error": str(error)})

    summary = {
        "timestamp": timestamp,
        "task": task_name,
        "models": models,
        "modes": modes,
        "results": all_results,
    }
    with (results_dir / "summary.json").open("w") as file:
        import json

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

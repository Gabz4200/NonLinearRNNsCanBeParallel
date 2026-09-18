"""Training script with fast-dev-run support."""

import sys
from pathlib import Path

# Normalize --fast-dev-run into Hydra-compatible assignment syntax.
if "--fast-dev-run" in sys.argv:
    idx = sys.argv.index("--fast-dev-run")
    if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
        value = sys.argv[idx + 1]
    else:
        value = "true"
    sys.argv[idx : idx + 2] = [f"fast_dev_run={value}"]

from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf


def _container(node: Any) -> dict[str, Any]:
    """Hydra node -> plain dict so downstream code stays registry-pluggable."""
    out = OmegaConf.to_container(node, resolve=True)
    assert isinstance(out, dict)
    return cast(dict[str, Any], dict(out))


# Ensure src imports work when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nonlinearrnnscanbeparallel.data.datamodule import GraphConnectivityDataModule  # noqa: E402
from nonlinearrnnscanbeparallel.models import M2RNN, MLPRNN, RKANRNN  # noqa: F401, E402
from nonlinearrnnscanbeparallel.modules.lightning_module import RNNTask  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import lightning as pl
    import torch

    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.runtime.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.runtime.matmul_precision)

    # Plain containers: the training path stays pluggable for any registered model.
    data_cfg = _container(cfg.get("data", {}))
    model_cfg = _container(cfg.get("model", {}))
    trainer_cfg = _container(cfg.get("trainer", {}))
    trainer_cfg["callbacks"] = _container(cfg.get("callbacks", {}))
    trainer_cfg["fast_dev_run"] = cfg.get("fast_dev_run", False)

    data_module = GraphConnectivityDataModule(data_cfg)

    # Model: any backbone registered via @register_model runs through this path.
    task = RNNTask(model_cfg)

    # Trainer: single construction path shared with engine.create_trainer.
    from nonlinearrnnscanbeparallel.training.engine import create_trainer

    trainer = create_trainer(trainer_cfg)

    trainer.fit(task, datamodule=data_module)


if __name__ == "__main__":
    main()

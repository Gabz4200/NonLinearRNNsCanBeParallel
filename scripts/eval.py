"""Evaluation script."""

import sys
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf


def _container(node: Any) -> dict[str, Any]:
    out = OmegaConf.to_container(node, resolve=True)
    assert isinstance(out, dict)
    return cast(dict[str, Any], dict(out))


# Ensure src imports work when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import lightning  # noqa: E402

import nonlinearrnnscanbeparallel.models  # noqa: F401, E402 - populates registry
from nonlinearrnnscanbeparallel.data.datamodule import GraphConnectivityDataModule  # noqa: E402
from nonlinearrnnscanbeparallel.models.registry import list_models  # noqa: E402
from nonlinearrnnscanbeparallel.modules.lightning_module import RNNTask  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    print("Registered models:", list_models())

    data_module = GraphConnectivityDataModule(_container(cfg.get("data", {})))
    data_module.setup("fit")  # creates train/val/test split

    task = RNNTask(_container(cfg.get("model", {})))

    trainer = lightning.Trainer(
        accelerator=cfg.trainer.get("accelerator", "auto"),
        devices=cfg.trainer.get("devices", 1),
        precision=cfg.trainer.get("precision", "32-true"),
        logger=False,
    )

    results = trainer.test(task, datamodule=data_module)
    print("Test results:", results)


if __name__ == "__main__":
    main()

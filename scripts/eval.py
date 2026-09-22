"""Evaluation script."""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Ensure src imports work when run as a script.
import lightning  # noqa: E402
from _util import node_dict  # noqa: E402

import nonlinearrnnscanbeparallel.models  # noqa: F401, E402 - populates registry (side effect)
from nonlinearrnnscanbeparallel.data.datamodule import GraphConnectivityDataModule  # noqa: E402
from nonlinearrnnscanbeparallel.models.registry import list_models  # noqa: E402
from nonlinearrnnscanbeparallel.modules.lightning_module import RNNTask  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    print("Registered models:", list_models())

    data_module = GraphConnectivityDataModule(node_dict(cfg.get("data", {})))
    data_module.setup("fit")  # creates train/val/test split

    task = RNNTask(node_dict(cfg.get("model", {})))

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

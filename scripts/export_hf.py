"""Export to HuggingFace format."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

import nonlinearrnnscanbeparallel.models  # noqa: F401 - populates MODEL_REGISTRY
from nonlinearrnnscanbeparallel.integrations.transformers import (
    NonLinearRNNsCanBeParallelConfig,
    NonLinearRNNsCanBeParallelModel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint to HF format")
    parser.add_argument("--checkpoint", required=True, help="Path to Lightning checkpoint")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    hparams = ckpt.get("hyper_parameters", {})
    model_cfg = hparams.get("model_cfg", {})

    config = NonLinearRNNsCanBeParallelConfig(
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_layers=model_cfg.get("num_layers", 4),
        num_heads=model_cfg.get("num_heads", 4),
        dropout=model_cfg.get("dropout", 0.1),
        model_name=model_cfg.get("name", "mlp_rnn"),
    )

    wrapper = NonLinearRNNsCanBeParallelModel(config)
    wrapper.native.load_state_dict(ckpt["state_dict"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    state_dict = wrapper.state_dict()
    save_file(state_dict, output_dir / "model.safetensors")

    print(f"Exported to {output_dir}")


if __name__ == "__main__":
    main()

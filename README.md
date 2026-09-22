# NonLinearRNNsCanBeParallel

PyTorch implementations of MLP-RNN, rKAN-RNN, M²RNN, minGRU, and minLSTM with a **chunkwise parallel training framework** for nonlinear RNNs, evaluated on deterministic graph-connectivity (USTCON-style) and language-modeling benchmarks.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

Recurrent models are inherently sequential at inference, but their training can be parallelized. This repo implements the "Nonlinear RNNs Can Be Parallel" framework (arXiv:2603.03612) and applies it to a family of nonlinear recurrent cells — including rKAN, matrix-state M²RNN, and the minimal log-space cells from the "Were RNNs All We Needed?" paper — using a scaffold/translator decomposition that trains in chunks while keeping O(1) sequential inference.

## Table of contents

- [Models](#models)
- [How parallel training works](#how-parallel-training-works)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Tasks](#tasks)
- [Repository layout](#repository-layout)
- [Development](#development)
- [Citations](#citations)

## Models

Each model implements the shared `RNNModule` protocol with a parallel `forward`, a single-token autoregressive `step`, and `init_state` for O(1) decoding. All use RMSNorm and SiLU activations at float32 precision.

| Model | Reference | Notes |
| --- | --- | --- |
| **MLP-RNN** | "Nonlinear RNNs Can Be Parallel" (arXiv:2603.03612, Def. 3, 7–10; Thm 1, Cor. 2) | Multi-head MLP recurrence, residual + FFN sublayers |
| **rKAN-RNN** | "rKAN: Rational Kolmogorov-Arnold Networks" (arXiv:2406.14495, Sec. 2–3, Eq. 9, 12) | Rational spline heads in the same multi-head recurrence |
| **M²RNN** | "M²RNN: Non-Linear RNNs with Matrix-Valued States" (arXiv:2603.14360, Sec. 3.1, Eq. 10–23) | Matrix-valued state `[N, K, V]` with causal conv cache |
| **minGRU** | "Were RNNs All We Needed?" (arXiv:2410.01201, Sec. 3.1, App. B) | Linear-time log-space parallel scan |
| **minLSTM** | "Were RNNs All We Needed?" (arXiv:2410.01201, Sec. 3.2, App. B) | Normalized gates in log-space |
| **Nano-RNN** | nanoGPT/nanoRWKV skeleton (see `models/nano_rnn.py`) | LM decoder; any cell above as the mixer |

## How parallel training works

The `ParallelRNNTrainer` wrapper (in `models/parallel_wrapper.py`) decomposes any target RNN into three parts per layer, so forward/backward can run over chunks instead of token-by-token:

1. **Scaffold** — a lightweight parallel-scannable cell (minGRU or minLSTM) summarizes each layer's input and produces boundary summaries at chunk starts.
2. **Translator** — maps those summaries into the target RNN's hidden-state space at chunk boundaries (MLP or rKAN, vector or matrix output).
3. **Target RNN** — runs in parallel across chunks from the translated boundary states, with in-chunk BPTT.

Gradients stay isolated: the target is trained from within-chunk BPTT, while scaffold and translator are trained from the boundary path. At inference the scaffold and translator are discarded — the target runs sequentially with `step`, so decoding cost is unchanged.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra cpu --extra dev
uv run python scripts/train.py --fast-dev-run true
uv run python scripts/eval.py --fast-dev-run true
```

> [!NOTE]
> `--extra cpu` installs CPU-only PyTorch. Use `--extra cu124` for CUDA 12.4. The two extras are mutually exclusive.

## Usage

Training is config-driven with [Hydra](https://hydra.cc). The task, the models in the run, and the training modes (parallel, BPTT, or both) are selected from config files or command-line overrides.

```bash
# default: sorted_graph_connectivity, mlp_rnn, parallel
uv run python scripts/train.py

# USTCON comparison: all models, parallel + BPTT
uv run python scripts/train.py --config-name config_ustcon

# language modeling on an OpenThoughts-114k subset
uv run python scripts/train.py --config-name config_lm

# tiny rKAN language model that fits a CPU-only box (~4 GB)
uv run python scripts/train.py --config-name config_lm_tiny

# single model / single mode, scaled down for a quick run
uv run python scripts/train.py --config-name config_ustcon \
  models=[mlp_rnn] modes=[parallel] \
  data.num_samples=64 data.max_seq_len=128 \
  model.hidden_dim=32 model.num_layers=1 \
  trainer.max_epochs=1
```

Each run writes a timestamped `results/<task>_<timestamp>/` directory with per-run loss curves, a validation-loss comparison plot, and a `summary.json` with metrics and wall-clock times.

Other entry points:

- `scripts/eval.py` — run the test split for the graph-connectivity task.
- `scripts/export_hf.py --checkpoint <ckpt> --output-dir <dir>` — export a Lightning checkpoint to Hugging Face `safetensors` format (see `integrations/transformers/`).
- `notebooks/01_quickstart.py` — jupytext notebook walking through model construction and forward passes.

## Tasks

| Task | Data module | Description |
| --- | --- | --- |
| `sorted_graph_connectivity` | `GraphConnectivityDataModule` | Predict 1 iff there is a path from source `s` to target `t`, given edges `(i, j)` in unary topological order (Def. 11, arXiv:2603.03612). |
| `ustcon` | `GraphReachabilityDataModule` | USTCON reachability on random graphs. |
| `long_sequence` | `LongSequenceLMDataModule` | Synthetic copy task with 16k-token sequences. |
| `openthoughts_lm` | `OpenThoughtsLMDataModule` | Causal LM over a tokenized OpenThoughts-114k subset. |

## Repository layout

```
src/nonlinearrnnscanbeparallel/
  models/       RNN cells, Nano-RNN, scaffold, translator, parallel wrapper
  modules/      Lightning tasks (BPTT + parallel paths)
  data/         task-specific Lightning data modules
  tasks/        task objectives, metrics, and gradient-clipping callbacks
  training/     trainer construction
  losses/       classification and language-modeling losses
  integrations/ transformers export (config + modeling)
configs/        Hydra configs (model, data, trainer, parallel, task presets)
scripts/        train / eval / export / param-count entry points
tests/          pytest suite
```

## Development

```bash
uv run pytest tests/
```

Formatting, linting, and type checks:

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pyrefly check
```

## Citations

```bibtex
@misc{merrill2026nonlinearrnns,
  title         = {Nonlinear RNNs Can Be Parallel},
  author        = {Merrill, William and Jiang, Hongjian and Li, Yanhong and Lin, Anthony and Sabharwal, Ashish},
  year          = {2026},
  eprint        = {2603.03612},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG}
}

@misc{aghaei2024rkan,
  title         = {rKAN: Rational Kolmogorov-Arnold Networks},
  author        = {Aghaei, Alireza Afzal and Hosseinzadeh, Mehdi and Parand, Kourosh},
  year          = {2024},
  eprint        = {2406.14495},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG}
}

@misc{mishra2026m2rnn,
  title         = {M²RNN: Non-Linear RNNs with Matrix-Valued States for Scalable Language Modeling},
  author        = {Mishra, Mayank and Tan, Shawn and Stoica, Ion and Gonzalez, Joseph E. and Dao, Tri},
  year          = {2026},
  eprint        = {2603.14360},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG}
}

@misc{feng2024rnns,
  title         = {Were RNNs All We Needed?},
  author        = {Feng, Leo and Tung, Frederick and Ahmed, Mohamed Osama and Bengio, Yoshua and Hajimirsadeghi, Hossein},
  year          = {2024},
  eprint        = {2410.01201},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG}
}
```
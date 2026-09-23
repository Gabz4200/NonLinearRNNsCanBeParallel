# NonLinearRNNsCanBeParallel

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Lightning](https://img.shields.io/badge/Lightning-2.3-792EE5?style=flat-square)](https://lightning.ai)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

Train nonlinear RNNs in parallel chunks without changing what the model learns.

[Overview](#overview) • [How it works](#how-parallel-training-works) • [Results](#results) • [Getting started](#getting-started) • [Usage](#usage) • [Tasks](#tasks) • [Repository layout](#repository-layout) • [Development](#development) • [FAQ](#faq) • [Citations](#citations)

This repo implements the "Nonlinear RNNs Can Be Parallel" framework (arXiv:2603.03612) and applies it to a family of nonlinear recurrent cells (MLP-RNN, rKAN, matrix-state M²RNN, and the minimal log-space cells from "Were RNNs All We Needed?"), evaluated on graph-connectivity benchmarks and causal language modeling.

## Overview

Recurrent models are sequential by nature at inference, but their *training* doesn't have to be. The `ParallelRNNTrainer` wrapper splits input sequences into chunks and trains all chunks simultaneously, using a small scaffold network to carry information across chunk boundaries. At inference the scaffold is discarded and the target RNN runs sequentially with O(1) per-token decoding, so decoding cost never changes.

The central claim under test here is **parity**: a model trained through the parallel path should learn exactly the same thing as the same model trained with plain BPTT. Every language-modeling config supports `modes=[parallel,bptt]` so you can verify that yourself in a single run.

## Models

Each model implements a shared protocol with a parallel `forward`, a single-token autoregressive `step`, and `init_state` for O(1) decoding.

| Model | Reference | Notes |
| --- | --- | --- |
| **MLP-RNN** | "Nonlinear RNNs Can Be Parallel" (arXiv:2603.03612) | Multi-head MLP recurrence, residual + FFN sublayers |
| **rKAN-RNN** | "rKAN" (arXiv:2406.14495) | Rational spline heads in the same multi-head recurrence |
| **M²RNN** | "M²RNN" (arXiv:2603.14360) | Matrix-valued state with causal conv cache |
| **minGRU / minLSTM** | "Were RNNs All We Needed?" (arXiv:2410.01201) | Log-space parallel scans; also used as scaffolds |
| **Nano-RNN** | nanoGPT/nanoRWKV skeleton | LM decoder; any cell above plugs in as the mixer |

## How parallel training works

`ParallelRNNTrainer` (`models/parallel_wrapper.py`) decomposes each layer into three parts, so forward and backward run over chunks instead of token-by-token:

1. Scaffold: a lightweight parallel-scannable cell (minGRU/minLSTM) that summarizes each layer's input into boundary states at chunk starts.
2. Translator: maps boundary summaries into the target RNN's hidden-state space (MLP or rKAN head).
3. Target RNN: runs in parallel across chunks from the translated boundary states, with in-chunk BPTT.

Gradients stay isolated: the target learns from within-chunk BPTT while scaffold and translator learn from the boundary path. If `chunk_size >= sequence length` the run falls back to a single chunk and the scaffold is skipped. The trainer prints a warning when that happens, since it means you're not actually training in parallel.

## Results

Parity check on real text: `nano_rnn` (rKAN mixer, 35M params) on 128 Wikipedia articles (EN + PT), 2 epochs on CPU, same data split for both modes.

| Metric | parallel | bptt |
| --- | --- | --- |
| val/loss | 5.9457 | 5.9488 |
| val/ppl | 489.8 | 502.0 |
| val/acc | 0.195 | 0.190 |
| NLL on unseen EN/PT sentences | 5.09 / 5.30 | 5.17 / 5.26 |

Validation loss tied to three decimals, and both modes score ~5.2 nats vs ~10.8 for a random-init control on held-out-style sentences. Per-run reports with metric-by-metric breakdowns live next to the numbers in `results/<task>_<timestamp>/RUN_REPORT.md`.

> [!NOTE]
> Quality parity is demonstrated; GPU throughput speedup is predicted but not yet measured. The most valuable next experiment is `config_wiki_small` on a GPU with both modes, logging tokens/sec alongside loss.

## Getting started

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra cpu --extra dev
uv run python scripts/train.py --config-name config_wiki_tiny fast_dev_run=true
```

> [!IMPORTANT]
> `--extra cpu` and `--extra cu124` (CUDA 12.4) are mutually exclusive. Pick one.

Verify the install with the test suite:

```bash
uv run pytest -q
```

## Usage

Training is config-driven with [Hydra](https://hydra.cc). Task, models, and modes (`parallel`, `bptt`, or both) come from the config files and can be overridden on the command line.

```bash
# default: sorted_graph_connectivity, mlp_rnn, parallel
uv run python scripts/train.py

# USTCON comparison: all models, parallel + BPTT
uv run python scripts/train.py --config-name config_ustcon

# Wikipedia LM, both modes in one run (parity check)
uv run python scripts/train.py --config-name config_wiki_tiny modes=[parallel,bptt]

# larger Wikipedia run for GPU boxes (parallel + BPTT)
uv run python scripts/train.py --config-name config_wiki_small

# tiny OpenThoughts LM for CPU-only boxes
uv run python scripts/train.py --config-name config_lm_tiny

# single model / single mode, scaled down
uv run python scripts/train.py --config-name config_ustcon \
  models=[mlp_rnn] modes=[parallel] \
  data.num_samples=64 data.max_seq_len=128 \
  model.hidden_dim=32 model.num_layers=1 \
  trainer.max_epochs=1
```

> [!TIP]
> Always smoke-test a new config first with `fast_dev_run=true` (a few batches) before committing to a full run.

Each run writes a timestamped `results/<task>_<timestamp>/` directory with the resolved `config.yaml`, per-run loss curves, a validation-loss comparison plot, and `summary.json` with metrics, checkpoints, and wall-clock times.

Other entry points:

- `scripts/eval.py`: test-split evaluation for the graph-connectivity task.
- `scripts/export_hf.py --checkpoint <ckpt> --output-dir <dir>`: export a Lightning checkpoint to Hugging Face `safetensors` format (see `integrations/transformers/`).
- `scripts/param_count.py`: parameter counts per model.
- `notebooks/01_quickstart.py`: jupytext notebook walking through model construction and forward passes.

## Tasks

| Task | Data module | Description |
| --- | --- | --- |
| `sorted_graph_connectivity` | `GraphConnectivityDataModule` | Path existence from `s` to `t` given edges in unary topological order |
| `ustcon` | `GraphReachabilityDataModule` | USTCON reachability on random graphs |
| `long_sequence` | `LongSequenceLMDataModule` | Synthetic copy task with 16k-token sequences |
| `openthoughts_lm` | `OpenThoughtsLMDataModule` | Causal LM over a tokenized OpenThoughts-114k subset |
| `wikipedia_lm` | `WikipediaLMDataModule` | Causal LM over EN + PT Wikipedia articles, cut into fixed blocks |

LM tasks log `train/loss`, `val/loss`, `val/nll`, `val/ppl`, and `val/acc`. Note `train/nll` and `val/nll` alias the cross-entropy loss (which already is the mean per-token NLL) under its own name.

## Repository layout

```
src/nonlinearrnnscanbeparallel/
  models/        RNN cells, Nano-RNN, scaffold, translator, parallel wrapper
  data/          task-specific Lightning data modules
  tasks/         LM + graph objectives, metrics, gradient-clipping callbacks
  losses/        classification and causal-LM losses
  logging/       NLL / perplexity / accuracy metrics
  training/      trainer construction
  modules/       legacy Lightning tasks (BPTT + parallel paths)
  integrations/  transformers export (config + modeling)
configs/         Hydra configs: model / data / trainer / parallel / task presets
scripts/         train / eval / export / param-count entry points
tests/           pytest suite (parity, causality, training smoke tests)
results/         timestamped run outputs + per-run reports
benchmarks/      throughput benchmarks
```

## Development

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pyrefly check
uv run pytest
```

Run focused tests while iterating, but run the full suite before delivery. New model code needs a reference-forward test on CPU, a parallel-vs-`step` consistency check, and, for anything touching the wrapper, a parallel-vs-BPTT parity case (see `tests/test_parallel_lm_parity.py`).

## FAQ

**Parallel and BPTT losses differ slightly. Is that expected?**
Small gaps (sub-percent) are normal: each mode re-initializes weights and shuffles data independently unless you fix a seed. Gaps at the third decimal of val loss and beyond deserve investigation.

**I see "single-chunk fallback" in the log. What does it mean?**
Your `parallel.chunk_size` is >= the sequence/block length, so the whole sequence is one chunk and the scaffold never engages. Lower `chunk_size` (e.g. 64 for 256-token blocks).

**Can I use the trained model for generation?**
Yes for experimentation: load the target weights from the checkpoint (`backbone.target.*` for parallel runs, `backbone.*` for BPTT) into a `NanoRNN` and decode with `step()`. At 35M params and a few epochs the output is collocations and loops, not usable text (see the sample reports in `results/`).

**Where do I add a new model?**
Implement the cell protocol in `models/`, register it in `models/registry.py`, add a `configs/model/` preset, and cover it in `tests/test_models.py`.

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

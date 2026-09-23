# AGENTS.md

## Project overview

Research codebase for **Nonlinear RNNs Can Be Parallel** (arXiv:2603.03612): train nonlinear recurrent models in parallel chunks (scaffold + translator + target RNN) with parity against plain BPTT, evaluated on graph-connectivity tasks and causal language modeling.

- **Stack**: Python 3.11+ (pinned `3.13.14`, upper bound `<3.14`), PyTorch 2.6, Lightning 2.3, Hydra/OmegaConf, Hugging Face `transformers`/`datasets`, uv, ruff, pyrefly, pytest.
- **Layout**: `src/nonlinearrnnscanbeparallel/` (models, data, tasks, losses, training, integrations), `configs/` (Hydra), `scripts/` (train/eval/export), `tests/`, `notebooks/` (jupytext `py:percent`), `results/` (run artifacts).
- **No monorepo, no CI workflows** (`.github/` absent). Pre-commit runs ruff + basic file hooks.

Human-facing docs live in `README.md`. This file is only what agents need to work here.

## Setup

Requires [uv](https://docs.astral.sh/uv/). CPU and CUDA extras are **mutually exclusive** — pick one:

```bash
uv sync --extra cpu --extra dev     # CPU boxes
uv sync --extra cu124 --extra dev   # CUDA 12.4 boxes
```

Optional: `--extra notebooks` for jupyter/jupytext. Verify install with `uv run pytest -q`.

After any `pyproject.toml` edit: `uv lock` then `uv lock --check`.

## Commands

### Training (Hydra)

```bash
# smoke-test a config first (few batches) — always do this for new/changed configs
uv run python scripts/train.py --config-name config_wiki_tiny fast_dev_run=true

# default config: sorted_graph_connectivity, mlp_rnn, parallel
uv run python scripts/train.py

# parity check: both modes in one run
uv run python scripts/train.py --config-name config_wiki_tiny modes=[parallel,bptt]

# dotted overrides for anything else
uv run python scripts/train.py --config-name config_ustcon \
  models=[mlp_rnn] modes=[parallel] data.num_samples=64 trainer.max_epochs=1
```

Other entry points:

- `uv run python scripts/eval.py` — graph-connectivity test-split eval.
- `uv run python scripts/export_hf.py --checkpoint <ckpt> --output-dir <dir>` — Lightning → HF safetensors. **Known broken** on transformers 5.x: `BaseModelOutput` must be imported from `transformers.modeling_outputs`, not the package root — fix before relying on export.
- `uv run python scripts/param_count.py` — param budget check for `config_lm` defaults; exits 1 if target is outside 124M–350M (not a pure printer).
- `notebooks/01_quickstart.py` — jupytext notebook; keep `py:percent` format (see `.jupytext.toml`).

Each run writes `results/<task>_<timestamp>/` (resolved config, plots, `summary.json`, `RUN_REPORT.md`) and Hydra logs under `outputs/`. Both are gitignored — never commit them.

### Quality gates (run before delivery, all from repo root)

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pyrefly check
uv run pytest
aislop scan .
```

- Focused iteration: `uv run pytest tests/test_parallel_wrapper.py` or `uv run pytest -k "<name>"`. Full suite before delivery (~80s, 120 tests).
- `aislop scan .` uses the global `aislop` binary (not a project dependency). Fix errors and fixable warnings; don't edit `.aislop/` config without user consent.
- **Known baseline**: `ruff check` currently reports ~34 pre-existing N806/N812/N803 findings (ML shape unpacking `B, T, D` and `import torch.nn.functional as F` vs pep8-naming), and `pyrefly check` reports ~14 pre-existing errors. Do not mass-rename unrelated code to silence them. Fix findings in code you touch; introduce no new ones.

### Pre-commit

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

Hooks: trailing whitespace, EOF, yaml check, large files, `ruff --fix`, `ruff-format`.

- **Known**: `pre-commit run --all-files` currently fails — pinned hook ruff `v0.6.9` differs from project ruff `0.16.7` (e.g. flags `UP038`, reformats some asserts). Prefer the Quality gates commands above for delivery; only chase pre-commit failures you introduce.

## Testing

- Location/convention: `tests/test_*.py`; `pythonpath = ["src"]`; default addopts `-v` (see `[tool.pytest.ini_options]` in `pyproject.toml`).
- Naming: `test_when_<scenario>_then_<outcome>`. Annotate test functions `-> None`. One Act per test. `pytest.raises` for expected exceptions; `@pytest.mark.parametrize` with descriptive `ids`.
- Mock only external boundaries; prefer real model code. No network, no sleeps, deterministic seeds.
- **New model code requires** (mirror existing patterns in `tests/test_models.py`):
  1. reference `forward` on CPU (shape + backward),
  2. full-sequence `forward` vs token-by-token `step` consistency,
  3. if touching `parallel_wrapper.py` or trainer paths: a parallel-vs-BPTT parity case (see `tests/test_parallel_lm_parity.py`).

## Code style

- Ruff: line length 100, target py311, rules `E,F,W,I,N,UP,B,C4,SIM`. Formatter: `uv run ruff format .`.
- Types: pyrefly (`pyrefly.toml`); covers `src/`, `tests/`, `scripts/`, `benchmarks/`. Excludes `.venv/`, `outputs/`, `notebooks/`, `integrations/`.
- Layout: package under `src/`; scripts stay thin and import from the package (they also `sys.path.insert` `src/` so plain `python scripts/...` works).
- Imports: stdlib → third-party → local; ruff isort rules enforce order.
- Fail fast: no silent fallbacks or broad `except` in library code; let programmer errors raise.
- Comments only for non-obvious intent/invariants (why a reshape, gradient-isolation reason, paper reference). No narrative comments, no commented-out code, no decorative separators.
- Tests and public docstrings use normal full-clarity prose.

## Architecture notes

### Model protocol

Every cell implements `BaseRNNModel` (`models/base.py`):

- `forward(x, state) -> (logits, new_state)` — full sequence `[B, T, D]`
- `step(x_t, state) -> (logits, new_state)` — single token `[B, D]`, O(1) decode
- `init_state(batch_size, device) -> RNNStateList`

State travels in `RNNState` / `RNNStateList` dataclasses (`hidden` + optional `extra`), not bare tensors. Keep mutation/cloning/detach explicit.

### Parallel training path

`models/parallel_wrapper.py` (`ParallelRNNTrainer`) decomposes each layer into scaffold (minGRU/minLSTM, parallel-scannable) → translator (MLP or rKAN) → target RNN (in-chunk BPTT). Gradients stay isolated across the boundary path. If `parallel.chunk_size >= sequence length`, run falls back to single-chunk sequential and warns — that is a config bug, not a feature; lower `chunk_size` (e.g. 64 for 256-token blocks).

### Adding a model

1. Implement the cell protocol in `src/nonlinearrnnscanbeparallel/models/`.
2. Decorate with `@register_model("<name>")` (`models/registry.py`).
3. Add a preset under `configs/model/<name>_<size>.yaml` and wire it into a `configs/config_*.yaml` `models:` list.
4. Cover it in `tests/test_models.py` (forward/backward + `step` parity).
5. `import nonlinearrnnscanbeparallel.models` has a **registry side effect** — needed in any entry point that calls `get_model`.

### Config system

Hydra configs compose `model` / `data` / `trainer` / `parallel` defaults (`configs/config.yaml` and `configs/config_*.yaml`). Top-level knobs: `task`, `models` (list), `modes` (`parallel` | `bptt` or both), `fast_dev_run`. Parallel knobs in `configs/parallel.yaml` (`chunk_size`, `scaffold_*`, `translator_*`). Prefer CLI dotted overrides over editing shared configs for one-off runs.

### Tasks and metrics

Tasks: `sorted_graph_connectivity`, `ustcon`, `long_sequence`, `openthoughts_lm`, `wikipedia_lm`. LM metrics: `train/loss`, `val/loss`, `val/nll`, `val/ppl`, `val/acc` (`val/nll` aliases cross-entropy). Central claim under test is **parity** between `parallel` and `bptt` — small sub-percent loss gaps are normal (independent weight init/shuffle unless seeded); third-decimal-and-beyond gaps need investigation.

## Build and deployment

No packaged deployment. Research loop only:

- Install/sync: `uv sync --extra cpu --extra dev` (or `cu124`).
- Export for HF consumers: `scripts/export_hf.py` → `safetensors` + config under `integrations/transformers/`.
- `benchmarks/` holds throughput benchmarks (currently empty scaffolding).

## PR / commit guidelines

- No required CI; local gates above are the bar. Run them before asking for review.
- Commit message: conventional style (`feat:`, `fix:`, `test:`, `chore:`, …), imperative mood, subject ≤50 chars when practical.
- Never commit: `.env`, secrets, `.venv`, `outputs/`, `lightning_logs/`, generated `results/*/`, `*.ckpt`, `.aislop/`, `.opencode/`, `.omp` (see `.gitignore`).
- Keep diffs scoped; don't reformat or rename unrelated code while fixing a bug.

## Gotchas

- `--extra cpu` and `--extra cu124` conflict — pick one or `uv sync` fails.
- `uv run aislop` is wrong (not a dependency); call `aislop scan .`.
- Scripts must run as `uv run python scripts/train.py ...` (Hydra `config_path=../configs` depends on that entry point).
- Parallel run checkpoint weights: `backbone.target.*` (parallel) vs `backbone.*` (BPTT).
- Generation from trained models is experimental only (collocations/loops at current scale) — load target weights into `NanoRNN` and decode with `step()`.
- Notebook edits: keep `notebooks/*.py` as jupytext percent scripts; don't hand-edit orphaned `.ipynb`.
- Don't run `pre-commit run --all-files` expecting a clean pass on untouched files (version skew with project ruff — see Pre-commit). It rewrites EOF newlines across many files; revert unrelated churn.

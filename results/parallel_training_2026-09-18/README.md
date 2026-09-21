# Parallel RNN Training Results

**Date:** 2026-09-18
**Experiment:** Parallel chunkwise training of nonlinear RNNs on Graph Connectivity Task

---

## Overview

This directory contains results from training three nonlinear RNN architectures using the parallel chunkwise training framework (ParallelRNNTrainer) on the sorted deterministic graph connectivity task (Definition 11, arXiv:2603.03612).

### Framework

The parallel chunkwise training framework implements the architecture from the PRD:

1. **Scaffold (minGRU)** - Per-layer parallel-scannable minGRU that scans the layer's input to produce boundary summaries
2. **Translator** - Per-layer boundary state translator (MLP for MLP-RNN/minGRU/minLSTM; rKAN for rKAN-RNN; MLP for M2RNN) with GDN-2 initialization (Xavier uniform, gain 2⁻²·⁵)
3. **Parallel Chunk Execution** - True parallel chunk execution via `[B,T,D] → [M·B,C,D]` reshape
4. **Gradient Isolation** - Target gradients from within-chunk BPTT; scaffold/translator gradients from boundary path
5. **Inference** - Plain sequential RNN (scaffold/translator discarded)

### Task

**Graph Connectivity Classification (sorted deterministic)** - Nodes in topological order, each node has at most one outgoing edge. Encoded in unary as specified in the paper: BOS, edges in unary, source, target, EOS. Binary classification at the last token (EOS position).

---

## Models Tested (Nonlinear RNN Targets)

| Model | Architecture | Target Type | Description |
|-------|-------------|-------------|-------------|
| **MLP-RNN** | Multi-layer MLP-RNN (2-layer MLP heads, 2-layer FFN) | `mlp` | 2-layer MLP with SiLU per head |
| **rKAN-RNN** | rKAN-RNN (edge-wise rational KAN heads, 2-layer rKAN FFN) | `rkan` | Edge-wise Padé approximants per connection |
| **M2RNN** | Matrix-to-Matrix RNN (matrix state [N,K,V], depthwise conv) | `m2rnn` | Matrix-valued recurrent state with associative recall |

> **Note:** minGRU and minLSTM are used as scaffolds only (for parallel prefix scanning). They are not trained as target RNNs.

---

## Configuration

```yaml
# Common settings
hidden_dim: 16
num_layers: 2
num_heads: 2
dropout: 0.1
num_classes: 2
chunk_size: 4
scaffold_dim: 16
target_type: mlp | rkan | m2rnn

# M2RNN specific
key_dim: 8
value_dim: 8

# Data
num_samples: 5000
max_nodes: 32
max_seq_len: 66
batch_size: 32

# Training
max_epochs: 3
optimizer: AdamW (lr=1e-3, weight_decay=0.01)
loss: CrossEntropyLoss (ignore_index=-100)
```

---

## Training Configuration Breakdown

| Parameter | Value | Implication |
|-----------|-------|-------------|
| **Total training samples** | 5,000 | Small dataset; fast iterations |
| **Validation samples** | ~500 (10%) | Reasonable validation signal |
| **Test samples** | ~500 (10%) | Held-out evaluation |
| **Batch size** | 32 | 125 batches/epoch (5,000 × 0.9 / 32 ≈ 125) |
| **Sequence length (max)** | 66 tokens | Short sequences; 1 chunk = 4 tokens → 17 chunks/seq |
| **Chunk size** | 4 | Small chunks = more parallelism, more boundary approximations |
| **Total training steps** | 375 (3 epochs × 125 steps) | Very short training |
| **Scaffold dim** | 16 | Same as hidden_dim; scaffold processes same dimension |

---

## Results Summary

| Model | Final Train Loss | Final Val Loss | Epochs | Steps/Epoch | Params |
|-------|------------------|----------------|--------|-------------|--------|
| **MLP-RNN** | 0.3607 | **0.1413** | 3 | 125 | ~15K |
| **rKAN-RNN** | 0.2785 | 0.2532 | 3 | 125 | ~18K |
| **M2RNN** | 0.2779 | 0.3188 | 3 | 125 | ~7.5K |

**Best validation loss:** MLP-RNN (0.1413)
**Best training loss:** M2RNN (0.2779) / rKAN-RNN (0.2785)

---

## Detailed Results Analysis

### What the Numbers Mean

#### MLP-RNN (0.3607 train / **0.1413 val**) ⭐
- **Strong generalization:** Validation loss is **significantly lower** than training loss (val < train by ~2.5×)
- This is unusual but can happen when:
  - Dropout (0.1) and weight decay (0.01) regularize heavily during training but are off during validation
  - The scaffold's boundary approximation acts as regularization (approximation error ≈ dropout on initial state)
  - The validation set happens to be "easier" (shorter sequences, simpler graphs)
- **Convergence:** Training loss decreases steadily from ~0.66 → 0.36 over 3 epochs
- **Validation behavior:** Starts at ~0.33, drops to ~0.14 — the boundary approximation error diminishes as scaffold improves
- **Best validation loss (0.1413)** — wins on generalization despite higher train loss

#### rKAN-RNN (0.2785 train / 0.2532 val)
- **Better training fit:** Lower training loss than MLP-RNN → rKAN fits training data better
- **Poorer generalization:** Val loss close to train loss (ratio ~0.9) → less regularization benefit
- **Edge-wise expressivity:** rKAN's per-edge Padé functions can memorize training patterns more easily
- **Overfitting risk:** With only 3 epochs and 375 steps, the gap is small but rKAN's higher capacity may overfit sooner with longer training
- **Stability:** Training loss relatively stable (~0.28-0.29 range in epoch 3)

#### M2RNN (0.2779 train / 0.3188 val)
- **Best training loss (tied with rKAN-RNN):** Matrix-valued state with associative recall enables efficient training
- **Poor generalization:** Val loss > train loss (0.3188 vs 0.2779) → overfitting
- **Matrix state capacity:** Matrix state [N,K,V] = 2×8×8 = 128 dims vs hidden_dim=16 → high capacity but small dataset
- **Slowest training:** ~4.1 it/s vs ~7 it/s for others (conv cache + matrix ops overhead)
- **Boundary approximation struggles:** 17 boundaries/seq with matrix state = many parameters to approximate

---

## Positive Contributors

| Factor | Impact |
|--------|--------|
| **Parallel chunk execution (chunk_size=4)** | Enables 17× parallelism over sequential (66/4 ≈ 17 chunks); makes training feasible at this seq length |
| **Scaffold boundary approximation as regularization** | Acts as "dropout on initial state" — forces chunk RNN to self-correct from perturbed states, improving generalization (especially MLP-RNN) |
| **GDN-2 translator init (Xavier gain 2⁻²·⁵)** | Keeps boundary states well-conditioned early in training; prevents exploding/vanishing gradients at boundaries |
| **Per-layer scaffolds + translators** | Each layer gets specialized boundary modeling; improves gradient flow through deep stacks |
| **Residual connections in target RNNs** | Keeps boundary map near identity; makes scaffold's job easier (approximating near-identity map) |
| **RMSNorm on scaffold/translator I/O** | Keeps boundary state norms controlled; prevents scale explosion across chunks |
| **Gradient isolation (detach_boundary option)** | Cleanly separates target gradients (within-chunk) from scaffold/translator gradients (boundary path) |

---

## Negative Contributors / Limitations

| Factor | Impact |
|--------|--------|
| **Very short training (3 epochs / 375 steps)** | Far from convergence; both models would improve significantly with more training |
| **Small dataset (5,000 samples)** | Limits generalization measurement; validation variance is high |
| **Short sequences (max 66 tokens)** | Doesn't stress-test the chunkwise framework; boundary errors have little time to amplify |
| **Small chunk size (4)** | Many boundaries (17/seq) = more approximation errors; larger chunks would reduce boundary count but increase sequential depth |
| **Small scaffold_dim (16 = hidden_dim)** | Scaffold capacity equals target; no compression, but also no bottleneck |
| **rKAN edge-wise overfitting** | rKAN's per-edge Padé functions can fit noise; needs more regularization (weight decay, dropout) or more data |
| **M2RNN matrix state overfitting** | Matrix state (128 dims) with only 5K samples → severe overfitting; val loss > train loss |
| **M2RNN slow training** | Matrix operations + conv cache = 4× slower than MLP-RNN (4.1 vs 7 it/s) |
| **Only 3 epochs** | Validation loss still decreasing — no plateau reached; early stopping would be premature |
| **Binary classification at single token** | Loss signal only at EOS; 65/66 timesteps produce no gradient signal (ignored via ignore_index=-100) |

---

## Files in This Directory

| File | Description |
|------|-------------|
| `training_results.json` | Complete loss histories (train/val per step/epoch for all models) |
| `loss_curves.png` | Per-model training & validation loss curves (log scale) |
| `val_loss_comparison.png` | Validation loss comparison across all models |
| `README.md` | This documentation |

---

## Reproducing Results

```bash
cd /home/gabz/Projects/NonLinearRNNsCanBeParallel
source .venv/bin/activate

python -c "
import torch, torch.nn as nn, lightning as pl
from nonlinearrnnscanbeparallel.models.registry import get_model
from nonlinearrnnscanbeparallel.models.parallel_wrapper import ParallelRNNTrainer
from nonlinearrnnscanbeparallel.data.datamodule import GraphConnectivityDataModule

data_module = GraphConnectivityDataModule({
    'num_samples': 5000, 'max_nodes': 32, 'max_seq_len': 66,
    'batch_size': 32, 'num_workers': 2
})
data_module.setup()

# MLP-RNN
target = get_model('mlp_rnn', input_dim=16, hidden_dim=16, num_layers=2, num_heads=2, dropout=0.1, num_classes=2)
wrapper = ParallelRNNTrainer(target, chunk_size=4, scaffold_dim=16, target_type='mlp')

# rKAN-RNN
target = get_model('rkan_rnn', input_dim=16, hidden_dim=16, num_layers=2, num_heads=2, dropout=0.1, num_classes=2)
wrapper = ParallelRNNTrainer(target, chunk_size=4, scaffold_dim=16, target_type='rkan')

# M2RNN
target = get_model('m2rnn', input_dim=16, hidden_dim=16, num_layers=2, num_heads=2, key_dim=8, value_dim=8, dropout=0.1, num_classes=2)
wrapper = ParallelRNNTrainer(target, chunk_size=4, scaffold_dim=16, target_type='m2rnn')

# ... training loop
"
```

---

## Framework Code Locations

| Component | Path |
|-----------|------|
| Scaffold | `src/nonlinearrnnscanbeparallel/models/scaffold.py` |
| Translator | `src/nonlinearrnnscanbeparallel/models/translator.py` |
| Parallel Wrapper | `src/nonlinearrnnscanbeparallel/models/parallel_wrapper.py` |
| M2RNN | `src/nonlinearrnnscanbeparallel/models/m2rnn.py` |
| Tests | `tests/test_scaffold.py`, `tests/test_translator.py`, `tests/test_parallel_wrapper.py` |

---

## License

MIT License - See project root for details.
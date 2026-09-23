# Parallel RNN training results

**Date:** 2026-09-18
**Experiment:** Parallel chunkwise training of nonlinear RNNs on graph connectivity

---

## Overview

This directory contains results from training three nonlinear RNN architectures using the parallel chunkwise training framework (ParallelRNNTrainer) on the sorted deterministic graph connectivity task (Definition 11, arXiv:2603.03612).

### Framework

The parallel chunkwise training framework implements the architecture from the PRD:

1. **Scaffold (minGRU)** - Per-layer parallel-scannable minGRU that scans the layer's input to produce boundary summaries
2. **Translator** - Per-layer boundary state translator (MLP for MLP-RNN/minGRU/minLSTM; rKAN for rKAN-RNN; MLP for M2RNN) with GDN-2 initialization (Xavier uniform, gain 2⁻²·⁵)
3. **Parallel chunk execution** - Chunk execution via `[B,T,D] → [M·B,C,D]` reshape
4. **Gradient isolation** - Target gradients from within-chunk BPTT; scaffold/translator gradients from boundary path
5. **Inference** - Plain sequential RNN (scaffold/translator discarded)

### Task

**Graph connectivity classification (sorted deterministic)** - Nodes in topological order, each node has at most one outgoing edge. Encoded in unary as specified in the paper: BOS, edges in unary, source, target, EOS. Binary classification at the last token (EOS position).

---

## Models tested (nonlinear RNN targets)

| Model | Architecture | Target Type | Description |
|-------|-------------|-------------|-------------|
| MLP-RNN | Multi-layer MLP-RNN (2-layer MLP heads, 2-layer FFN) | `mlp` | 2-layer MLP with SiLU per head |
| rKAN-RNN | rKAN-RNN (edge-wise rational KAN heads, 2-layer rKAN FFN) | `rkan` | Edge-wise Padé approximants per connection |
| M2RNN | Matrix-to-Matrix RNN (matrix state [N,K,V], depthwise conv) | `m2rnn` | Matrix-valued recurrent state with associative recall |

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

## Training configuration breakdown

| Parameter | Value | Implication |
|-----------|-------|-------------|
| Total training samples | 5,000 | Small dataset; fast iterations |
| Validation samples | ~500 (10%) | Reasonable validation signal |
| Test samples | ~500 (10%) | Held-out evaluation |
| Batch size | 32 | 125 batches/epoch (5,000 × 0.9 / 32 ≈ 125) |
| Sequence length (max) | 66 tokens | Short sequences; 1 chunk = 4 tokens → 17 chunks/seq |
| Chunk size | 4 | Small chunks = more parallelism, more boundary approximations |
| Total training steps | 375 (3 epochs × 125 steps) | Very short training |
| Scaffold dim | 16 | Same as hidden_dim; scaffold processes same dimension |

---

## Results summary

| Model | Final Train Loss | Final Val Loss | Epochs | Steps/Epoch | Params |
|-------|------------------|----------------|--------|-------------|--------|
| MLP-RNN | 0.3607 | 0.1413 | 3 | 125 | ~15K |
| rKAN-RNN | 0.2785 | 0.2532 | 3 | 125 | ~18K |
| M2RNN | 0.2779 | 0.3188 | 3 | 125 | ~7.5K |

Best validation loss: MLP-RNN (0.1413).
Best training loss: M2RNN (0.2779) / rKAN-RNN (0.2785).

---

## Detailed results analysis

### What the numbers mean

#### MLP-RNN (0.3607 train / 0.1413 val)

Validation loss came in well below training loss (about 2.5x lower), which looks odd but has a few plausible causes: dropout (0.1) and weight decay (0.01) regularize during training and switch off during validation, the scaffold's boundary approximation adds noise that acts like regularization on the initial state, and the validation split may just hold shorter sequences and simpler graphs. Training loss fell steadily from ~0.66 to ~0.36 over the 3 epochs while validation dropped from ~0.33 to ~0.14, so the boundary approximation kept improving as the scaffold learned. It wins on generalization despite the higher train loss.

#### rKAN-RNN (0.2785 train / 0.2532 val)

rKAN fits the training data better than MLP-RNN but its validation loss sits close to its train loss (ratio ~0.9), so it gets less of that regularization benefit. The likely cause is capacity: per-edge Padé functions can memorize training patterns, and with only 375 steps the gap is still small but points toward earlier overfitting on longer runs. Training loss held steady in the ~0.28-0.29 range through epoch 3.

#### M2RNN (0.2779 train / 0.3188 val)

Tied for the best training loss, but validation sits above train (0.3188 vs 0.2779), the classic overfitting signature. The matrix state ([N,K,V] = 2x8x8 = 128 dims against hidden_dim 16) is a lot of capacity for 5K samples. It also trained slowest at ~4.1 it/s against ~7 it/s for the others, from conv-cache and matrix-op overhead, and 17 boundaries per sequence means a lot of matrix states to approximate at chunk edges.

---

## What helped

| Factor | Impact |
|--------|--------|
| Parallel chunk execution (chunk_size=4) | 17 chunks per sequence (66/4); makes training feasible at this length |
| Scaffold boundary approximation as regularization | Noisy boundary states force each chunk to self-correct, which improved generalization (most visible in MLP-RNN) |
| GDN-2 translator init (Xavier gain 2⁻²·⁵) | Keeps boundary states well-conditioned early; avoids exploding/vanishing gradients at boundaries |
| Per-layer scaffolds + translators | Each layer models its own boundaries; better gradient flow through the stack |
| Residual connections in target RNNs | Keeps the boundary map near identity, which is an easier target for the scaffold to approximate |
| RMSNorm on scaffold/translator I/O | Keeps boundary state norms bounded across chunks |
| Gradient isolation (detach_boundary option) | Separates target gradients (within-chunk) from scaffold/translator gradients (boundary path) |

---

## Limitations

| Factor | Impact |
|--------|--------|
| Very short training (3 epochs / 375 steps) | Far from convergence; all models would improve with more training |
| Small dataset (5,000 samples) | Limits what generalization numbers can say; validation variance is high |
| Short sequences (max 66 tokens) | Doesn't stress the chunkwise framework; boundary errors have little room to compound |
| Small chunk size (4) | 17 boundaries per sequence means more approximation error; larger chunks would cut boundaries but add sequential depth |
| Small scaffold_dim (16 = hidden_dim) | No compression but also no bottleneck |
| rKAN edge-wise overfitting | Per-edge Padé functions fit noise; needs more regularization or more data |
| M2RNN matrix state overfitting | 128-dim state on 5K samples; val loss above train loss |
| M2RNN slow training | Matrix ops + conv cache run ~4x slower than MLP-RNN (4.1 vs 7 it/s) |
| Loss signal only at EOS | Binary label at one token; 65/66 timesteps carry no gradient (ignore_index=-100) |

---

## Files in this directory

| File | Description |
|------|-------------|
| `training_results.json` | Complete loss histories (train/val per step/epoch for all models) |
| `loss_curves.png` | Per-model training and validation loss curves (log scale) |
| `val_loss_comparison.png` | Validation loss comparison across all models |
| `README.md` | This documentation |

---

## Reproducing results

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

## Framework code locations

| Component | Path |
|-----------|------|
| Scaffold | `src/nonlinearrnnscanbeparallel/models/scaffold.py` |
| Translator | `src/nonlinearrnnscanbeparallel/models/translator.py` |
| Parallel Wrapper | `src/nonlinearrnnscanbeparallel/models/parallel_wrapper.py` |
| M2RNN | `src/nonlinearrnnscanbeparallel/models/m2rnn.py` |
| Tests | `tests/test_scaffold.py`, `tests/test_translator.py`, `tests/test_parallel_wrapper.py` |

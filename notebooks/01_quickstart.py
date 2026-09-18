# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quickstart: NonLinearRNNsCanBeParallel
#
# This notebook demonstrates the MLP-RNN and rKAN-RNN implementations
# with the sorted deterministic graph connectivity task (Definition 11,
# arXiv:2603.03612).

# %%
import torch

from nonlinearrnnscanbeparallel.models.registry import get_model, list_models

print("Registered models:", list_models())

# %%
# Create MLP-RNN.
model = get_model("mlp_rnn", hidden_dim=128, num_layers=2, num_heads=4)
x = torch.randn(4, 16, 128)
out, states = model(x)
print(f"MLP-RNN output shape: {out.shape}")
print(f"Number of recurrent states: {len(states)}")

# %%
# Create rKAN-RNN.
rkan_model = get_model("rkan_rnn", hidden_dim=128, num_layers=2, num_heads=4)
out2, states2 = rkan_model(x)
print(f"rKAN-RNN output shape: {out2.shape}")
print(f"Number of recurrent states: {len(states2)}")

# %%
# Test with token IDs as used in training.
from nonlinearrnnscanbeparallel.modules.lightning_module import RNNTask  # noqa: E402

task_cfg = {
    "name": "mlp_rnn",
    "hidden_dim": 128,
    "num_layers": 2,
    "num_heads": 4,
    "vocab_size": 36,
}
task = RNNTask(task_cfg)

# Input batch of token IDs [B, T].
token_ids = torch.randint(0, 36, (2, 10))
logits, states = task(token_ids)
print(f"Task logits shape: {logits.shape}")
print(f"States: {len(states)}")

# %%
# Run forward and backward.
loss = logits.sum()
loss.backward()
print("Backward pass successful")

# %%
# Exercise the data module.
from nonlinearrnnscanbeparallel.data.datamodule import GraphConnectivityDataModule  # noqa: E402

data_cfg = {
    "task": "sorted_graph_connectivity",
    "max_nodes": 16,
    "num_samples": 100,
    "batch_size": 8,
    "max_seq_len": 34,
}
data_module = GraphConnectivityDataModule(data_cfg)
data_module.setup("fit")

batch = next(iter(data_module.train_dataloader()))
print(f"Batch input_ids shape: {batch['input_ids'].shape}")
print(f"Batch labels shape: {batch['labels'].shape}")
print(f"Sample labels: {batch['labels'][0]}")

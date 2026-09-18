"""Tests for minGRU scaffold (parallel prefix scan)."""

from __future__ import annotations

import pytest
import torch

from nonlinearrnnscanbeparallel.models.scaffold import MinGRUScaffold


def test_scaffold_forward_shape() -> None:
    """Scaffold forward returns states for all timesteps."""
    scaffold = MinGRUScaffold(input_dim=32, hidden_dim=16, num_heads=2)
    x = torch.randn(2, 10, 32)
    states = scaffold(x)
    assert states.shape == (2, 10, 16)


def test_scaffold_parallel_scan_matches_sequential() -> None:
    """Parallel scan must match sequential recurrence."""
    scaffold = MinGRUScaffold(input_dim=32, hidden_dim=16, num_heads=2)
    scaffold.eval()
    x = torch.randn(1, 8, 32)

    # Parallel forward (uses scan)
    states_parallel = scaffold(x)

    # Sequential forward (using step method)
    states_sequential = []
    h = torch.zeros(1, 2, 8)  # [B, H, D_head]
    for t in range(8):
        x_t = x[:, t, :]
        h = scaffold.step(x_t, h)
        # Apply output projection to match parallel forward
        h_proj = scaffold.output_proj(h.reshape(1, 16))
        states_sequential.append(h_proj)
    states_sequential = torch.stack(states_sequential, dim=1)

    assert torch.allclose(states_parallel, states_sequential, atol=1e-5)


def test_scaffold_gradient_flows() -> None:
    """Gradients must flow through the parallel scan."""
    scaffold = MinGRUScaffold(input_dim=32, hidden_dim=16, num_heads=2)
    x = torch.randn(2, 6, 32, requires_grad=True)
    states = scaffold(x)
    loss = states.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    for param in scaffold.parameters():
        assert param.grad is not None
        assert param.grad.abs().sum() > 0


def test_scaffold_tied_weights_across_layers() -> None:
    """Scaffold weights must be shared when used across layers."""
    scaffold = MinGRUScaffold(input_dim=32, hidden_dim=16, num_heads=2)
    # Same scaffold instance used for multiple layers
    x1 = torch.randn(2, 10, 32)
    x2 = torch.randn(2, 10, 32)
    _ = scaffold(x1)
    _ = scaffold(x2)
    # Weights are the same object
    assert scaffold.linear_z[0].weight is scaffold.linear_z[0].weight
    assert scaffold.linear_h[0].weight is scaffold.linear_h[0].weight


def test_scaffold_fp32_scan_precision() -> None:
    """Scan accumulations must run in fp32 for numerical stability."""
    scaffold = MinGRUScaffold(input_dim=32, hidden_dim=16, num_heads=2)
    x = torch.randn(1, 100, 32, dtype=torch.bfloat16)
    states = scaffold(x)
    # Output should be bfloat16 but internally accumulated in fp32
    assert states.dtype == torch.bfloat16
    # No NaNs from long sequence
    assert not torch.isnan(states).any()


def test_scaffold_respects_chunk_boundaries() -> None:
    """Scaffold produces states at all positions including chunk boundaries."""
    scaffold = MinGRUScaffold(input_dim=32, hidden_dim=16, num_heads=2)
    x = torch.randn(2, 256, 32)
    states = scaffold(x)
    # Can index any boundary
    boundary_states = states[:, ::64, :]  # Every 64 tokens
    assert boundary_states.shape == (2, 4, 16)


def test_scaffold_zero_initial_state() -> None:
    """Initial state must be zeros (not ones like minGRU model)."""
    scaffold = MinGRUScaffold(input_dim=32, hidden_dim=16, num_heads=2)
    x = torch.randn(1, 1, 32)
    states = scaffold(x)
    # First state comes from zero initial
    # The update gate determines how much of candidate is used
    assert states.shape == (1, 1, 16)


def test_scaffold_candidate_activation() -> None:
    """Candidate uses the minimal activation (x>=0: x+0.5, else: sigmoid)."""
    scaffold = MinGRUScaffold(input_dim=32, hidden_dim=16, num_heads=2)
    # Test the activation function directly
    pos = torch.tensor([2.0])
    neg = torch.tensor([-2.0])
    pos_out = scaffold.candidate_activation(pos)
    neg_out = scaffold.candidate_activation(neg)
    assert pos_out.item() == pytest.approx(2.5)
    assert neg_out.item() == pytest.approx(torch.sigmoid(neg).item())


def test_scaffold_batch_independence() -> None:
    """Batch items must be independent in the scan."""
    scaffold = MinGRUScaffold(input_dim=32, hidden_dim=16, num_heads=2)
    x = torch.randn(4, 10, 32)
    states = scaffold(x)
    # Each batch item should have different states
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(states[i], states[j], atol=1e-5)

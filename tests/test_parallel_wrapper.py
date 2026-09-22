"""Tests for Parallel RNN Training Wrapper."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from nonlinearrnnscanbeparallel.models.parallel_wrapper import ParallelRNNTrainer
from nonlinearrnnscanbeparallel.models.registry import get_model
from nonlinearrnnscanbeparallel.models.scaffold import MinGRUScaffold, ScaffoldStack
from nonlinearrnnscanbeparallel.models.translator import TranslatorLayer, TranslatorStack


def test_parallel_wrapper_forward_shape() -> None:
    """Wrapper processes full sequence and returns logits for all timesteps."""
    target = get_model(
        "mlp_rnn",
        input_dim=32,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    wrapper = ParallelRNNTrainer(target, chunk_size=8, scaffold_dim=16)

    x = torch.randn(2, 32, 32)  # 32 timesteps, 4 chunks of 8
    logits = wrapper(x)
    assert logits.shape == (2, 32, 2)


def test_parallel_wrapper_chunked_forward() -> None:
    """Wrapper splits sequence into chunks and processes in parallel."""

    target = get_model(
        "mlp_rnn",
        input_dim=32,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    wrapper = ParallelRNNTrainer(target, chunk_size=4, scaffold_dim=16)

    x = torch.randn(1, 12, 32)  # 12 timesteps, 3 chunks of 4
    logits = wrapper(x)
    assert logits.shape == (1, 12, 2)


def test_parallel_wrapper_scaffold_per_layer() -> None:
    """Each target layer has its own scaffold."""

    target = get_model(
        "mlp_rnn",
        input_dim=32,
        hidden_dim=32,
        num_layers=3,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    wrapper = ParallelRNNTrainer(target, chunk_size=8, scaffold_dim=16)

    # Check per-layer scaffolds
    assert hasattr(wrapper, "scaffolds")
    assert len(wrapper.scaffolds) == 3
    # Each scaffold stack is a separate instance
    for i in range(3):
        for j in range(i + 1, 3):
            left = cast(ScaffoldStack, wrapper.scaffolds[i]).layers[0]
            right = cast(ScaffoldStack, wrapper.scaffolds[j]).layers[0]
            assert isinstance(left, MinGRUScaffold)
            assert isinstance(right, MinGRUScaffold)
            assert left.linear_z[0].weight is not right.linear_z[0].weight


def test_parallel_wrapper_translator_per_layer() -> None:
    """Each target layer has its own translator."""

    target = get_model(
        "mlp_rnn",
        input_dim=32,
        hidden_dim=32,
        num_layers=3,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    wrapper = ParallelRNNTrainer(target, chunk_size=8, scaffold_dim=16)

    assert len(wrapper.translators) == 3
    # Each translator stack is a separate instance
    for i in range(3):
        for j in range(i + 1, 3):
            left = cast(TranslatorStack, wrapper.translators[i]).layers[0]
            right = cast(TranslatorStack, wrapper.translators[j]).layers[0]
            assert isinstance(left, TranslatorLayer)
            assert isinstance(right, TranslatorLayer)
            left_net = cast(torch.nn.Sequential, left.net)
            right_net = cast(torch.nn.Sequential, right.net)
            assert left_net[0].weight is not right_net[0].weight


def test_parallel_wrapper_gradient_isolation() -> None:
    """Target gradients come only from within-chunk BPTT; scaffold/translator from boundary."""

    target = get_model(
        "mlp_rnn",
        input_dim=32,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    wrapper = ParallelRNNTrainer(target, chunk_size=4, scaffold_dim=16)

    x = torch.randn(2, 8, 32, requires_grad=True)
    logits = wrapper(x)
    loss = logits.sum()
    loss.backward()

    # Target parameters used in forward should get gradients
    # (emb_norm is not used - wrapper uses input_proj instead)
    used_params = [p for n, p in target.named_parameters() if "emb_norm" not in n]
    for param in used_params:
        assert param.grad is not None, "Expected gradient for target param"
    for param in wrapper.scaffolds[0].parameters():
        assert param.grad is not None
    for trans in wrapper.translators:
        for param in trans.parameters():
            assert param.grad is not None


def test_parallel_wrapper_boundary_approximation() -> None:
    """Boundary states from translator approximate true hidden states."""

    target = get_model(
        "mlp_rnn",
        input_dim=32,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    wrapper = ParallelRNNTrainer(target, chunk_size=4, scaffold_dim=16)

    x = torch.randn(1, 8, 32)
    # Get boundary states from wrapper
    boundary_states = wrapper.get_boundary_states(x)
    # Returns list per layer
    assert len(boundary_states) == 1
    assert boundary_states[0].shape == (1, 1, 32)  # 1 boundary for 2 chunks


def test_parallel_wrapper_inference_mode() -> None:
    """In inference, wrapper runs target RNN sequentially without scaffold."""

    target = get_model(
        "mlp_rnn",
        input_dim=32,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    wrapper = ParallelRNNTrainer(target, chunk_size=4, scaffold_dim=16)

    wrapper.eval()
    x = torch.randn(1, 8, 32)
    logits = wrapper(x)
    assert logits.shape == (1, 8, 2)


def test_parallel_wrapper_step_matches_sequential() -> None:
    """Autoregressive step() must match sequential forward in inference."""
    target = get_model(
        "mlp_rnn",
        input_dim=32,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    wrapper = ParallelRNNTrainer(target, chunk_size=4, scaffold_dim=16)
    wrapper.eval()

    x = torch.randn(1, 6, 32)
    # Full forward
    logits = wrapper(x)

    # Step by step
    step_logits = []
    state = None
    for t in range(6):
        out_t, state = wrapper.step(x[:, t, :], state)
        step_logits.append(out_t)
    rollout = torch.stack(step_logits, dim=1)

    assert torch.allclose(logits, rollout, atol=1e-5)


def test_parallel_wrapper_different_chunk_sizes() -> None:
    """Wrapper works with different chunk sizes."""

    for chunk_size in [2, 4, 8, 16]:
        target = get_model(
            "mlp_rnn",
            input_dim=16,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            num_classes=2,
        )
        wrapper = ParallelRNNTrainer(target, chunk_size=chunk_size, scaffold_dim=8)

        seq_len = chunk_size * 3
        x = torch.randn(1, seq_len, 16)
        logits = wrapper(x)
        assert logits.shape == (1, seq_len, 2)


def test_when_single_chunk_then_scaffold_skipped_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """chunk_size >= T warns and runs the target sequentially without the scaffold."""
    target = get_model(
        "mlp_rnn",
        input_dim=16,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    wrapper = ParallelRNNTrainer(target, chunk_size=8, scaffold_dim=8)
    wrapper.train()

    def _boom(x: torch.Tensor) -> torch.Tensor:
        raise AssertionError("scaffold must not run on a single chunk")

    monkeypatch.setattr(wrapper.scaffolds[0], "forward", _boom)

    x = torch.randn(1, 8, 16)
    with pytest.warns(UserWarning, match="single-chunk"):
        logits = wrapper(x)
    assert logits.shape == (1, 8, 2)


def test_when_parallel_then_first_chunk_starts_zeroed() -> None:
    """Chunk 0 is not conditioned on scaffold/translator (zeroed start); later chunks are."""
    target = get_model(
        "mlp_rnn",
        input_dim=16,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    wrapper = ParallelRNNTrainer(target, chunk_size=4, scaffold_dim=8, dropout=0.0)
    wrapper.train()
    x = torch.randn(2, 12, 16)  # 3 chunks of 4

    baseline = wrapper(x)

    def _constant_boundaries(z: torch.Tensor) -> torch.Tensor:
        return torch.full((z.shape[0], z.shape[1], 16), 42.0)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(wrapper.translators[0], "forward", _constant_boundaries)
    try:
        constant = wrapper(x)
    finally:
        monkeypatch.undo()

    # Chunk 0 runs from the zeroed state, so it must not see the boundary path.
    torch.testing.assert_close(constant[:, :4, :], baseline[:, :4, :])
    # Later chunks start from translated boundaries, so they must differ.
    assert not torch.allclose(constant[:, 4:, :], baseline[:, 4:, :], atol=1e-6)


def test_when_parallel_then_boundaries_causal_over_full_prefix() -> None:
    """Scanner state at t depends only on inputs [0..t] and early inputs reach far chunks."""
    target = get_model(
        "mlp_rnn",
        input_dim=16,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )
    wrapper = ParallelRNNTrainer(target, chunk_size=4, scaffold_dim=8, dropout=0.0)
    scaffold = wrapper.scaffolds[0]
    x = torch.randn(1, 12, 16)
    base = scaffold(x)

    late = x.clone()
    late[:, 8:] += 1.0  # perturb inside the last chunk only
    assert torch.allclose(scaffold(late)[:, :8], base[:, :8], atol=1e-5)

    early = x.clone()
    early[:, 0] += 3.0 * torch.sign(torch.randn(1, 16))  # non-uniform perturb of the first token
    assert (scaffold(early)[:, 7] - base[:, 7]).abs().max() > 1e-5


def test_parallel_wrapper_target_types() -> None:
    """Wrapper works with different target RNN types."""
    for model_name, target_type in [
        ("mlp_rnn", "mlp"),
        ("rkan_rnn", "rkan"),
        ("min_gru", "mlp"),
        ("min_lstm", "mlp"),
    ]:
        if model_name in ["min_gru", "min_lstm"]:
            target = get_model(
                model_name,
                input_dim=16,
                hidden_dim=16,
                num_layers=1,
                num_heads=2,
                dropout=0.0,
                num_classes=2,
            )
        else:
            target = get_model(
                model_name,
                input_dim=16,
                hidden_dim=16,
                num_layers=1,
                num_heads=2,
                dropout=0.0,
                num_classes=2,
            )

        wrapper = ParallelRNNTrainer(target, chunk_size=4, scaffold_dim=8, target_type=target_type)
        x = torch.randn(1, 8, 16)
        logits = wrapper(x)
        assert logits.shape == (1, 8, 2)

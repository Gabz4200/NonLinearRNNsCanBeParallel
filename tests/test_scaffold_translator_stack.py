"""Tests for scaffold and translator stacks (depth + type selection)."""

from __future__ import annotations

import pytest
import torch

from nonlinearrnnscanbeparallel.models.base import RMSNorm
from nonlinearrnnscanbeparallel.models.scaffold import (
    FourierScaffold,
    LinearSumScaffold,
    MinGRUScaffold,
    MinLSTMScaffold,
    ScaffoldStack,
)
from nonlinearrnnscanbeparallel.models.translator import TranslatorLayer, TranslatorStack


def test_when_scaffold_stack_depth_one_then_shape_matches_legacy() -> None:
    stack = ScaffoldStack(
        scaffold_type="min_gru",
        input_dim=16,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        positional=False,
    )
    legacy = MinGRUScaffold(input_dim=16, hidden_dim=8, num_heads=2, dropout=0.0)
    stack.layers[0].load_state_dict(legacy.state_dict())
    x = torch.randn(2, 6, 16)
    torch.testing.assert_close(stack(x), legacy(x))


def test_when_scaffold_positional_default_then_stack_adds_pe() -> None:
    pe_on = ScaffoldStack(scaffold_type="linear_sum", input_dim=16, hidden_dim=8, num_layers=1)
    pe_off = ScaffoldStack(
        scaffold_type="linear_sum", input_dim=16, hidden_dim=8, num_layers=1, positional=False
    )
    pe_off.layers[0].load_state_dict(pe_on.layers[0].state_dict())
    x = torch.randn(2, 6, 16)
    assert pe_on.positional is True
    assert not torch.allclose(pe_on(x), pe_off(x))


def test_when_fourier_scaffold_then_output_shape_and_backward() -> None:
    scaffold = FourierScaffold(input_dim=16, hidden_dim=8)
    x = torch.randn(2, 6, 16, requires_grad=True)
    out = scaffold(x)
    assert out.shape == (2, 6, 8)
    out.sum().backward()
    assert x.grad is not None
    assert scaffold.input_proj.weight.grad is not None


def test_when_fourier_scaffold_is_causal_then_future_perturbation_does_not_change_prefix() -> None:
    scaffold = FourierScaffold(input_dim=16, hidden_dim=8)
    scaffold.eval()
    x = torch.randn(2, 6, 16)
    with torch.no_grad():
        baseline = scaffold(x)
        perturbed = x.clone()
        perturbed[:, 4:, :] += 100.0  # change future rows only
        changed = scaffold(perturbed)
    torch.testing.assert_close(baseline[:, :4, :], changed[:, :4, :])
    assert not torch.allclose(baseline[:, 4:, :], changed[:, 4:, :])


def test_when_min_lstm_scaffold_then_output_shape() -> None:
    scaffold = MinLSTMScaffold(input_dim=16, hidden_dim=8, num_heads=2, dropout=0.0)
    states = scaffold(torch.randn(2, 6, 16))
    assert states.shape == (2, 6, 8)


def test_when_min_lstm_scaffold_step_then_matches_forward() -> None:
    scaffold = MinLSTMScaffold(input_dim=16, hidden_dim=8, num_heads=2, dropout=0.0).eval()
    x = torch.randn(1, 5, 16)
    parallel = scaffold(x)
    h = torch.zeros(1, 2, 4)
    steps = []
    for t in range(5):
        h = scaffold.step(x[:, t, :], h)
        steps.append(scaffold.output_proj(h.reshape(1, 8)))
    sequential = torch.stack(steps, dim=1)
    torch.testing.assert_close(parallel, sequential, rtol=1e-4, atol=1e-5)


def test_when_scaffold_stack_depth_three_then_output_shape() -> None:
    stack = ScaffoldStack(
        scaffold_type="min_gru", input_dim=16, hidden_dim=8, num_layers=3, num_heads=2, dropout=0.0
    )
    assert len(stack.layers) == 3
    assert stack(torch.randn(2, 6, 16)).shape == (2, 6, 8)


def test_when_linear_sum_scaffold_then_matches_linspace_reference() -> None:
    scaffold = LinearSumScaffold(input_dim=16, hidden_dim=8, min_weight=0.1)
    x = torch.randn(2, 6, 16)
    out = scaffold(x)
    proj = x @ scaffold.input_proj.weight.T
    reference = torch.zeros_like(proj)
    for t in range(1, proj.shape[1] + 1):
        weights = torch.linspace(0.1, 1.0, t)
        reference[:, t - 1] = (proj[:, :t] * weights.view(1, t, 1)).sum(dim=1)
    torch.testing.assert_close(out, reference)
    assert out.shape == (2, 6, 8)
    out.sum().backward()
    assert scaffold.input_proj.weight.grad is not None


def test_when_linear_sum_scaffold_single_position_then_min_weight_scaled() -> None:
    scaffold = LinearSumScaffold(input_dim=16, hidden_dim=8, min_weight=0.1)
    x = torch.randn(2, 1, 16)
    out = scaffold(x).squeeze(1)
    expected = 0.1 * (x @ scaffold.input_proj.weight.T).squeeze(1)
    torch.testing.assert_close(out, expected)


def test_when_linear_sum_scaffold_stack_depth_three_then_output_shape() -> None:
    stack = ScaffoldStack(scaffold_type="linear_sum", input_dim=16, hidden_dim=8, num_layers=3)
    assert len(stack.layers) == 3
    assert all(isinstance(layer, LinearSumScaffold) for layer in stack.layers)
    assert stack(torch.randn(2, 6, 16)).shape == (2, 6, 8)


def test_when_linear_sum_scaffold_invalid_min_weight_then_raises() -> None:
    with pytest.raises(ValueError, match="min_weight must be in"):
        LinearSumScaffold(input_dim=16, hidden_dim=8, min_weight=0.0)
    with pytest.raises(ValueError, match="min_weight must be in"):
        LinearSumScaffold(input_dim=16, hidden_dim=8, min_weight=1.0)


def test_when_translator_pre_norm_disabled_then_identity() -> None:
    default = TranslatorLayer(input_dim=12, hidden_dim=16, translator_type="mlp")
    off = TranslatorLayer(input_dim=12, hidden_dim=16, translator_type="mlp", pre_norm=False)
    assert isinstance(default.input_norm, RMSNorm)
    assert isinstance(off.input_norm, torch.nn.Identity)


def test_when_translator_norm_type_layernorm_then_used() -> None:
    layer = TranslatorLayer(
        input_dim=12, hidden_dim=16, translator_type="mlp", norm_type="layernorm"
    )
    assert isinstance(layer.input_norm, torch.nn.LayerNorm)
    stack = TranslatorStack(
        input_dim=12, hidden_dim=16, translator_type="mlp", norm_type="layernorm"
    )
    assert isinstance(stack.layers[0].input_norm, torch.nn.LayerNorm)
    assert stack(torch.randn(2, 4, 12)).shape == (2, 4, 16)


def test_when_translator_invalid_norm_type_then_raises() -> None:
    with pytest.raises(ValueError, match="Unknown norm_type"):
        TranslatorLayer(input_dim=12, hidden_dim=16, translator_type="mlp", norm_type="batchnorm")


def test_when_scaffold_stack_invalid_type_then_raises() -> None:
    with pytest.raises(ValueError, match="Unknown scaffold_type"):
        ScaffoldStack(scaffold_type="gru", input_dim=16, hidden_dim=8)


def test_when_translator_stack_depth_one_then_matches_single_layer_mlp() -> None:
    stack = TranslatorStack(input_dim=12, hidden_dim=16, translator_type="mlp", num_layers=1)
    legacy = TranslatorLayer(input_dim=12, hidden_dim=16, translator_type="mlp")
    stack.eval()
    legacy.eval()
    stack.layers[0].load_state_dict(legacy.state_dict())
    x = torch.randn(2, 4, 12)
    torch.testing.assert_close(stack(x), legacy(x))


def test_when_translator_stack_depth_three_then_output_shape() -> None:
    stack = TranslatorStack(input_dim=12, hidden_dim=16, translator_type="mlp", num_layers=3)
    assert len(stack.layers) == 3
    assert stack(torch.randn(2, 4, 12)).shape == (2, 4, 16)


def test_when_translator_rkan_type_then_output_shape() -> None:
    layer = TranslatorLayer(input_dim=12, hidden_dim=16, translator_type="rkan")
    assert layer(torch.randn(2, 4, 12)).shape == (2, 4, 16)


def test_when_translator_matrix_shape_then_only_final_layer_reshapes() -> None:
    stack = TranslatorStack(
        input_dim=12,
        hidden_dim=16,
        translator_type="mlp",
        num_layers=3,
        output_shape="matrix",
        num_heads=2,
        key_dim=4,
        value_dim=4,
    )
    assert stack(torch.randn(2, 4, 12)).shape == (2, 4, 2, 4, 4)


def test_when_translator_stack_invalid_depth_then_raises() -> None:
    with pytest.raises(ValueError, match="num_layers must be >= 1"):
        TranslatorStack(input_dim=12, hidden_dim=16, num_layers=0)


def test_when_translator_invalid_type_then_raises() -> None:
    with pytest.raises(ValueError, match="Unknown translator_type"):
        TranslatorLayer(input_dim=12, hidden_dim=16, translator_type="conv")

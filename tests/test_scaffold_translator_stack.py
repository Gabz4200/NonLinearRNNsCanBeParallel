"""Tests for scaffold and translator stacks (depth + type selection)."""

from __future__ import annotations

import pytest
import torch

from nonlinearrnnscanbeparallel.models.scaffold import (
    MinGRUScaffold,
    MinLSTMScaffold,
    ScaffoldStack,
)
from nonlinearrnnscanbeparallel.models.translator import (
    Translator,
    TranslatorLayer,
    TranslatorStack,
)


def test_when_scaffold_stack_depth_one_then_shape_matches_legacy() -> None:
    stack = ScaffoldStack(
        scaffold_type="min_gru", input_dim=16, hidden_dim=8, num_layers=1, num_heads=2, dropout=0.0
    )
    legacy = MinGRUScaffold(input_dim=16, hidden_dim=8, num_heads=2, dropout=0.0)
    stack.layers[0].load_state_dict(legacy.state_dict())
    x = torch.randn(2, 6, 16)
    torch.testing.assert_close(stack(x), legacy(x))


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


def test_when_scaffold_stack_invalid_type_then_raises() -> None:
    with pytest.raises(ValueError, match="Unknown scaffold_type"):
        ScaffoldStack(scaffold_type="gru", input_dim=16, hidden_dim=8)


def test_when_translator_stack_depth_one_then_matches_legacy_mlp() -> None:
    stack = TranslatorStack(input_dim=12, hidden_dim=16, translator_type="mlp", num_layers=1)
    legacy = Translator(input_dim=12, hidden_dim=16, target_type="mlp")
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

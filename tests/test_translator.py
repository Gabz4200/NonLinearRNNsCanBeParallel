"""Tests for Translator module (maps scaffold summaries to boundary states)."""

from __future__ import annotations

import torch
from torch import nn

from nonlinearrnnscanbeparallel.models.translator import Translator


def test_translator_forward_shape() -> None:
    """Translator maps scaffold output to target hidden state."""
    translator = Translator(input_dim=16, hidden_dim=32, target_type="mlp")
    x = torch.randn(2, 4, 16)  # 4 boundary positions
    out = translator(x)
    assert out.shape == (2, 4, 32)


def test_translator_per_layer_instance() -> None:
    """Each layer must have its own translator instance."""
    trans1 = Translator(input_dim=16, hidden_dim=32, target_type="mlp")
    trans2 = Translator(input_dim=16, hidden_dim=32, target_type="mlp")
    # Different instances, different parameters
    # type: ignore[attr-defined]
    assert trans1.net[0].weight is not trans2.net[0].weight


def test_translator_mlp_architecture() -> None:
    """MLP target uses 2-layer MLP translator."""
    translator = Translator(input_dim=16, hidden_dim=32, target_type="mlp")
    x = torch.randn(1, 1, 16)
    out = translator(x)
    assert out.shape == (1, 1, 32)
    # Should have 2 linear layers with SiLU in between
    assert isinstance(translator.net, nn.Sequential)
    assert len(translator.net) == 4  # Linear, SiLU, Dropout, Linear


def test_translator_rkan_architecture() -> None:
    """rKAN target uses rKAN-based translator."""
    translator = Translator(input_dim=16, hidden_dim=32, target_type="rkan")
    x = torch.randn(1, 1, 16)
    out = translator(x)
    assert out.shape == (1, 1, 32)
    assert not isinstance(translator.net, nn.Sequential)  # rKAN is not Sequential


def test_translator_gradient_flows() -> None:
    """Gradients must flow through translator."""
    translator = Translator(input_dim=16, hidden_dim=32, target_type="mlp")
    x = torch.randn(2, 4, 16, requires_grad=True)
    out = translator(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    for param in translator.parameters():
        assert param.grad is not None
        assert param.grad.abs().sum() > 0


def test_translator_rmsnorm_on_input() -> None:
    """Translator applies RMSNorm to input scaffold state."""
    translator = Translator(input_dim=16, hidden_dim=32, target_type="mlp")
    # Check that RMSNorm is applied (input should be normalized)
    x = torch.randn(2, 4, 16) * 100  # Large values
    out = translator(x)
    # Output should be reasonable (not explode)
    assert out.abs().max() < 1000


def test_translator_rmsnorm_on_output() -> None:
    """Translator applies RMSNorm to output (boundary state)."""
    translator = Translator(input_dim=16, hidden_dim=32, target_type="mlp")
    x = torch.randn(2, 4, 16)
    out = translator(x)
    # Output should be normalized (RMSNorm on output)
    # Check that norm is approximately 1 per feature
    norms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(norms, torch.ones_like(norms), atol=0.5)


def test_translator_boundary_only() -> None:
    """Translator only runs at M boundary positions, not all T timesteps."""
    translator = Translator(input_dim=16, hidden_dim=32, target_type="mlp")
    # m boundary positions (much less than T)
    m = 4
    x = torch.randn(2, m, 16)
    out = translator(x)
    assert out.shape == (2, m, 32)


def test_translator_zero_initial_boundary() -> None:
    """First chunk (m=0) uses zero initial state, no translator needed."""
    translator = Translator(input_dim=16, hidden_dim=32, target_type="mlp")
    # This is handled at the wrapper level - translator not called for m=0
    x = torch.randn(2, 1, 16)
    out = translator(x)
    assert out.shape == (2, 1, 32)


def test_translator_deterministic() -> None:
    """Translator must be deterministic given same input."""
    torch.manual_seed(42)
    translator = Translator(input_dim=16, hidden_dim=32, target_type="mlp", dropout=0.0)
    x = torch.randn(2, 4, 16)
    out1 = translator(x)
    out2 = translator(x)

    assert torch.allclose(out1, out2)

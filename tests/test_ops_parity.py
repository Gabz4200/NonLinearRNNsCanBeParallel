"""Parity tests for the accelerated RNN-step backends.

The reference op in :mod:`nonlinearrnnscanbeparallel.ops.rnn_step` is the
correctness oracle. The Taichi kernel is a separate implementation of the same
recurrence, so the only thing that makes it safe to select is a test asserting
the two agree numerically.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nonlinearrnnscanbeparallel.models.base import RNNState
from nonlinearrnnscanbeparallel.models.mlp_rnn import MultiHeadRNNLayer
from nonlinearrnnscanbeparallel.models.registry import get_model
from nonlinearrnnscanbeparallel.ops.rnn_step import rnn_step_reference
from nonlinearrnnscanbeparallel.ops.rnn_step_impls import ReferenceOp

ti = pytest.importorskip("taichi", reason="the taichi extra is not installed")


@pytest.fixture(scope="module", autouse=True)
def _taichi_runtime() -> None:
    from nonlinearrnnscanbeparallel.kernels.taichi.runtime import ensure_initialized

    ensure_initialized("cpu")


def _layer(hidden_dim: int = 32, num_heads: int = 4) -> MultiHeadRNNLayer:
    return MultiHeadRNNLayer(hidden_dim, num_heads, dropout=0.1).eval()


class _Head(nn.Module):
    """A head with an arbitrary Linear/SiLU topology, mirroring MLPHead's shape.

    MLPHead keeps its stack in ``self.net`` and concatenates ``[h_prev, x_t]``
    before it; the kernel walks exactly that structure, so this test double
    reproduces the contract rather than the class.
    """

    def __init__(self, head_dim: int, linears: list[tuple[int, bool]]) -> None:
        super().__init__()
        self.head_dim = head_dim
        modules: list[nn.Module] = []
        in_dim = head_dim * 2
        for index, (out_dim, bias) in enumerate(linears):
            if index:
                modules.append(nn.SiLU())
            modules.append(nn.Linear(in_dim, out_dim, bias=bias))
            in_dim = out_dim
        self.net = nn.Sequential(*modules)

    def forward(self, h_prev: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h_prev, x_t], dim=-1))


def _layer_with_topology(
    hidden_dim: int, num_heads: int, linears: list[tuple[int, bool]]
) -> MultiHeadRNNLayer:
    """A layer whose heads have the given per-layer (out_features, has_bias) shape."""
    return MultiHeadRNNLayer(
        hidden_dim, num_heads, dropout=0.1, head_factory=lambda d: _Head(d, linears)
    ).eval()


def _assert_matches(layer: MultiHeadRNNLayer, hidden_dim: int, num_heads: int) -> None:
    from nonlinearrnnscanbeparallel.kernels.taichi.rnn_step import rnn_step as ti_step

    head_dim = hidden_dim // num_heads
    torch.manual_seed(11)
    x = torch.randn(3, hidden_dim)
    h_prev = torch.randn(3, num_heads, head_dim)

    reference = rnn_step_reference(x, h_prev, layer)
    accelerated = ti_step(x, h_prev, layer)

    torch.testing.assert_close(accelerated, reference, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "linears",
    [
        pytest.param([(8, False)], id="single-linear-no-activation"),
        pytest.param([(16, False), (16, False), (8, False)], id="three-linear"),
        pytest.param([(8, False)] * 4, id="four-linear-uniform-width"),
        pytest.param([(64, False), (8, False)], id="wide-then-narrow"),
        pytest.param([(4, False), (8, False), (8, False)], id="narrow-then-wide"),
    ],
)
def test_when_head_depth_or_width_differs_then_kernel_still_matches(
    linears: list[tuple[int, bool]],
) -> None:
    _assert_matches(_layer_with_topology(32, 4, linears), 32, 4)


@pytest.mark.parametrize(
    ("bias_first", "bias_rest"),
    [
        pytest.param(True, True, id="bias-on-every-layer"),
        pytest.param(False, True, id="bias-on-trailing-layers-only"),
        pytest.param(True, False, id="bias-on-leading-layer-only"),
    ],
)
def test_when_head_has_biases_then_kernel_still_matches(bias_first: bool, bias_rest: bool) -> None:
    """Bias handling is a separate code path: it is read out of the weight block."""
    linears = [(16, bias_first), (16, bias_rest), (8, bias_rest)]
    _assert_matches(_layer_with_topology(32, 4, linears), 32, 4)


class _GeluHead(_Head):
    """A head whose activation is not SiLU, so the kernel must refuse it."""

    def __init__(self, head_dim: int) -> None:
        super().__init__(head_dim, [(head_dim, False), (head_dim, False)])
        self.net = nn.Sequential(
            nn.Linear(head_dim * 2, head_dim, bias=False),
            nn.GELU(),
            nn.Linear(head_dim, head_dim, bias=False),
        )


class _NarrowInputHead(_Head):
    """A head whose first layer consumes fewer than 2*head_dim inputs."""

    def __init__(self, head_dim: int) -> None:
        super().__init__(head_dim, [(head_dim, False), (head_dim, False)])
        self.net = nn.Sequential(
            nn.Linear(head_dim, head_dim, bias=False),
            nn.SiLU(),
            nn.Linear(head_dim, head_dim, bias=False),
        )


def test_when_head_is_not_a_linear_silu_stack_then_kernel_rejects_it() -> None:
    """An unsupported topology must raise, not silently compute something else."""
    from nonlinearrnnscanbeparallel.kernels.taichi.rnn_step import rnn_step as ti_step

    layer = MultiHeadRNNLayer(32, 4, dropout=0.1, head_factory=_GeluHead).eval()
    x = torch.randn(2, 32)
    h_prev = torch.randn(2, 4, 8)

    with pytest.raises(ValueError, match="Linear, SiLU"):
        ti_step(x, h_prev, layer)


def test_when_head_geometry_is_wrong_then_kernel_rejects_it() -> None:
    from nonlinearrnnscanbeparallel.kernels.taichi.rnn_step import rnn_step as ti_step

    layer = MultiHeadRNNLayer(32, 4, dropout=0.1, head_factory=_NarrowInputHead).eval()
    x = torch.randn(2, 32)
    h_prev = torch.randn(2, 4, 8)

    with pytest.raises(ValueError, match=r"expected 2\*head_dim"):
        ti_step(x, h_prev, layer)


@pytest.mark.parametrize(
    ("hidden_dim", "num_heads", "batch"),
    [
        pytest.param(32, 4, 5, id="four-heads-batch-five"),
        pytest.param(16, 2, 1, id="two-heads-batch-one"),
        pytest.param(64, 8, 3, id="eight-heads-batch-three"),
    ],
)
def test_when_taichi_kernel_runs_then_it_matches_the_reference(
    hidden_dim: int, num_heads: int, batch: int
) -> None:
    from nonlinearrnnscanbeparallel.kernels.taichi.rnn_step import rnn_step as ti_step

    torch.manual_seed(0)
    layer = _layer(hidden_dim, num_heads)
    head_dim = hidden_dim // num_heads
    x = torch.randn(batch, hidden_dim)
    h_prev = torch.randn(batch, num_heads, head_dim)

    reference = rnn_step_reference(x, h_prev, layer)
    accelerated = ti_step(x, h_prev, layer)

    assert accelerated.shape == reference.shape
    torch.testing.assert_close(accelerated, reference, rtol=1e-5, atol=1e-5)


def test_when_taichi_kernel_runs_twice_then_it_is_deterministic() -> None:
    from nonlinearrnnscanbeparallel.kernels.taichi.rnn_step import rnn_step as ti_step

    torch.manual_seed(1)
    layer = _layer()
    x = torch.randn(4, 32)
    h_prev = torch.randn(4, 4, 8)

    first = ti_step(x, h_prev, layer)
    second = ti_step(x, h_prev, layer)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_when_layer_is_training_then_taichi_kernel_refuses() -> None:
    """The kernel cannot reproduce the reference's dropout, so it must not pretend to."""
    from nonlinearrnnscanbeparallel.kernels.taichi.rnn_step import rnn_step as ti_step

    torch.manual_seed(2)
    layer = MultiHeadRNNLayer(32, 4, dropout=0.5).train()
    x = torch.randn(2, 32)
    h_prev = torch.randn(2, 4, 8)

    with pytest.raises(RuntimeError, match="inference-only"):
        ti_step(x, h_prev, layer)


def test_when_head_state_is_flat_then_taichi_kernel_still_matches() -> None:
    """A [B, H*D_head] state is accepted, as the layer's own step accepts it."""
    from nonlinearrnnscanbeparallel.kernels.taichi.rnn_step import rnn_step as ti_step

    torch.manual_seed(3)
    layer = _layer()
    x = torch.randn(3, 32)
    h_flat = torch.randn(3, 32)

    reference = rnn_step_reference(x, h_flat, layer)
    accelerated = ti_step(x, h_flat, layer)

    torch.testing.assert_close(accelerated, reference, rtol=1e-5, atol=1e-5)


def test_when_input_shape_is_wrong_then_taichi_kernel_rejects_it() -> None:
    from nonlinearrnnscanbeparallel.kernels.taichi.rnn_step import rnn_step as ti_step

    torch.manual_seed(4)
    layer = _layer()

    with pytest.raises(ValueError, match=r"x must be \[batch, 32\]"):
        ti_step(torch.randn(2, 16), torch.randn(2, 4, 8), layer)


def test_when_module_is_not_a_recurrent_layer_then_both_paths_reject_it() -> None:
    from nonlinearrnnscanbeparallel.kernels.taichi.rnn_step import rnn_step as ti_step

    with pytest.raises(TypeError):
        rnn_step_reference(torch.randn(2, 8), torch.randn(2, 2, 4), nn.Linear(8, 8))

    with pytest.raises(TypeError):
        ti_step(torch.randn(2, 8), torch.randn(2, 2, 4), nn.Linear(8, 8))


def test_when_backend_is_selected_then_dispatch_routes_to_the_kernel() -> None:
    """build_op('taichi') must not silently fall back to the reference."""
    from nonlinearrnnscanbeparallel.ops.rnn_step_impls import BackendOp, build_op

    assert isinstance(build_op("taichi"), BackendOp)
    assert isinstance(build_op("reference"), ReferenceOp)


def test_when_registry_builds_a_mlp_rnn_then_its_layer_feeds_the_kernel() -> None:
    """The kernel must work on a layer built through the public registry path."""
    from nonlinearrnnscanbeparallel.kernels.taichi.rnn_step import rnn_step as ti_step

    torch.manual_seed(5)
    model = get_model(
        "nano_rnn",
        input_dim=32,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        mixer_type="mlp",
        vocab_size=16,
    ).eval()
    layer = model.layers[0][0]

    x = torch.randn(2, 32)
    h_prev = torch.randn(2, 4, 8)

    reference = rnn_step_reference(x, h_prev, layer)
    accelerated = ti_step(x, h_prev, layer)

    torch.testing.assert_close(accelerated, reference, rtol=1e-5, atol=1e-5)


def test_when_reference_runs_on_a_layer_then_it_equals_the_layer_step() -> None:
    """The oracle must itself agree with the layer, or parity proves nothing."""
    torch.manual_seed(6)
    layer = _layer()
    x = torch.randn(4, 32)
    h_prev = torch.randn(4, 4, 8)

    reference = rnn_step_reference(x, h_prev, layer)
    layer_state = layer.step(x, RNNState(hidden=h_prev))[1].hidden

    torch.testing.assert_close(reference, layer_state, rtol=1e-5, atol=1e-5)

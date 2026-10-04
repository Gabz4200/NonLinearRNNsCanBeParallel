"""Taichi RNN step kernel for the multi-head MLP-RNN layer.

Recomputes the same recurrence as
:func:`nonlinearrnnscanbeparallel.ops.rnn_step.rnn_step_reference`, which is
the correctness oracle; ``tests/test_ops_parity.py`` asserts the two agree.

Scope
-----
Supports :class:`~nonlinearrnnscanbeparallel.models.mlp_rnn.MultiHeadRNNLayer`
heads built as an alternating ``Linear``/``SiLU`` stack, at any depth and any
width, with or without biases -- which covers
:class:`~nonlinearrnnscanbeparallel.models.mlp_rnn.MLPHead` in all its
configured shapes.  A head containing anything else (a normalisation, a
different nonlinearity) raises rather than silently computing a different
function.

The math is written out instead of reusing the layer because the layer stacks
per-head weights into one batched ``ParameterDict`` and runs the heads through
``torch.vmap``.  A Taichi kernel wants explicit per-head tensors, and doing that
reshuffle here keeps the vmap knowledge at the boundary.  Depth and width are
baked into the compiled kernel because Taichi requires compile-time-constant
loop bounds; kernels are cached against that geometry.

Inference only
--------------
Dropout is deliberately absent.  A Taichi kernel cannot draw the same mask as
the reference for a given RNG state, so including it would make the two
irreconcilable.  Callers must pass a layer in ``eval()``; this raises on a
training-mode layer instead of quietly returning an unmasked result that
silently disagrees with the reference.
"""

from typing import Any, cast

import torch
from torch import nn

from .runtime import ensure_initialized

# Taichi compiles per signature and per dtype, so both are keyed in the cache
# together; recompiling on every call would cost more than the kernel saves.
_KERNEL_CACHE: dict[tuple[Any, ...], Any] = {}


def _head_topology(layer: nn.Module) -> list[tuple[int, int, bool]]:
    """Return the per-layer (in_features, out_features, has_bias) of the head.

    Walks the meta skeleton in order, so any depth, any hidden width, and
    optional biases are described rather than assumed.

    Raises:
        TypeError: the module is not a multi-head RNN layer.
        ValueError: the head is not a plain Linear stack of the form
            ``[Linear, act, (act, Linear)*]`` -- i.e. the concatenation of
            ``[h_prev, x_t]`` followed by Linear layers with SiLU in between.
            Dropout layers are tolerated and skipped, since the kernel is
            inference-only and never applies them.
    """
    try:
        heads = list(layer.heads)  # type: ignore[attr-defined]
        int(layer.num_heads)  # type: ignore[attr-defined]
        int(layer.head_dim)  # type: ignore[attr-defined]
        int(layer.hidden_dim)  # type: ignore[attr-defined]
    except AttributeError as error:
        raise TypeError(
            "the Taichi rnn_step kernel requires a MultiHeadRNNLayer-shaped module, "
            f"got {type(layer).__name__}"
        ) from error

    if not heads:
        raise ValueError("layer has no heads")

    net = heads[0].net  # type: ignore[attr-defined]
    linears: list[nn.Linear] = []
    expect_linear = True
    for module in net:
        if isinstance(module, nn.Linear):
            if not expect_linear:
                raise ValueError(
                    "the Taichi rnn_step kernel expects alternating Linear/SiLU layers, "
                    f"found two Linear layers with no activation between ({type(module).__name__})"
                )
            linears.append(module)
            expect_linear = False
        elif isinstance(module, nn.SiLU):
            if expect_linear:
                raise ValueError(
                    "the Taichi rnn_step kernel expects the head to start with a Linear layer, "
                    "found SiLU first"
                )
            expect_linear = True
        elif isinstance(module, (nn.Dropout, nn.Identity)):
            continue  # inference-only: dropout is never applied
        else:
            raise ValueError(
                "the Taichi rnn_step kernel supports only Linear, SiLU, and Dropout "
                f"layers in a head, found {type(module).__name__}"
            )

    if expect_linear or not linears:
        raise ValueError("the Taichi rnn_step kernel expects the head to end with a Linear layer")

    return [(int(m.in_features), int(m.out_features), m.bias is not None) for m in linears]


def _head_weights(
    layer: nn.Module, topology: list[tuple[int, int, bool]]
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Extract stacked head weights as [H, out, in], plus per-layer biases.

    The layer stores heads batched, so the weight tensors arrive already
    leading-dimensioned by head. Keys are matched by module path rather than a
    fixed stride, because whether a bias exists changes which indices are
    present.
    """
    params = cast("dict[str, torch.Tensor]", layer.params)
    weight_keys = sorted(
        (k for k in params if k.endswith("/weight")), key=lambda k: int(k.split("/")[1])
    )
    if not weight_keys:
        raise ValueError("head has no Linear weights")

    weights: list[torch.Tensor] = []
    biases: list[torch.Tensor | None] = []
    for layer_index, (_, _, has_bias) in enumerate(topology):
        key = weight_keys[layer_index]
        weights.append(params[key].detach())
        bias_key = key[: -len("weight")] + "bias"
        biases.append(params[bias_key].detach() if has_bias and bias_key in params else None)
    return weights, biases


def _build_kernel(
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
    depth: int,
    width: int,
    has_bias: bool,
) -> Any:
    """Compile kernels for a head of a given depth and padded width.

    Taichi traces the kernel body and cannot subscript a captured Python list
    with a loop index, so the per-layer geometry is not read from ``topology``
    here. Instead every layer runs the same ``width``-wide loops over a
    zero-padded weight block, which is exact: a padded weight column
    contributes nothing to the sum, and a padded activation is only ever read
    through such a zero column, so it cannot reach the output. ``width``
    therefore takes the max over the layer's in/out features and
    ``2 * head_dim``, which keeps the concatenated-input reads in bounds.
    """
    import taichi as ti

    # Taichi reads these annotations itself: a kernel argument must be declared
    # with the call form ti.types.ndarray(...), which a type checker flags as an
    # invalid annotation. The suppression is on each line because that is the
    # only place it applies. This module also omits
    # `from __future__ import annotations`, which would stringify the calls and
    # make Taichi reject them.
    @ti.kernel
    def norm_kernel(
        # pyrefly: ignore [invalid-annotation]
        x: ti.types.ndarray(dtype=ti.f32, ndim=2),  # [batch, hidden_dim]
        # pyrefly: ignore [invalid-annotation]
        norm_w: ti.types.ndarray(dtype=ti.f32, ndim=1),  # [hidden_dim]
        norm_eps: float,
        # pyrefly: ignore [invalid-annotation]
        x_norm: ti.types.ndarray(dtype=ti.f32, ndim=3),  # [batch, num_heads, head_dim]
    ):
        # The RMS reduction must not share a kernel body with the per-head
        # output loop: interleaving the two made this kernel disagree with the
        # reference by ~0.7 absolute, consistently and at every batch size.
        # Running the reduction on its own (batch, hidden) iteration space and
        # materialising x_norm brings it to ~1e-7. Verified both ways in
        # tests/test_ops_parity.py.
        batch = x.shape[0]
        for b, i in ti.ndrange(batch, hidden_dim):
            sq_sum = 0.0
            for k in range(hidden_dim):
                v = x[b, k]
                sq_sum += v * v
            inv = 1.0 / (sq_sum / hidden_dim + norm_eps) ** 0.5
            x_norm[b, i // head_dim, i % head_dim] = x[b, i] * inv * norm_w[i]

    @ti.kernel
    def step_kernel(
        # pyrefly: ignore [invalid-annotation]
        x_norm: ti.types.ndarray(dtype=ti.f32, ndim=3),  # [batch, num_heads, head_dim]
        # pyrefly: ignore [invalid-annotation]
        h: ti.types.ndarray(dtype=ti.f32, ndim=3),  # [batch, num_heads, head_dim]
        # pyrefly: ignore [invalid-annotation]
        weights: ti.types.ndarray(dtype=ti.f32, ndim=4),  # [depth, num_heads, max_width, max_width]
        # pyrefly: ignore [invalid-annotation]
        biases: ti.types.ndarray(dtype=ti.f32, ndim=3),  # [depth, num_heads, max_width]
        # pyrefly: ignore [invalid-annotation]
        scratch: ti.types.ndarray(dtype=ti.f32, ndim=3),  # [depth, batch, max_width]
        # pyrefly: ignore [invalid-annotation]
        out: ti.types.ndarray(dtype=ti.f32, ndim=3),  # [batch, num_heads, head_dim]
    ):
        batch = x_norm.shape[0]
        for b, hd in ti.ndrange(batch, num_heads):
            for layer_index in range(depth):
                for j in range(width):
                    total = biases[layer_index, hd, j] if has_bias else 0.0
                    for k in range(width):
                        # The first layer reads the concatenation [h_prev, x_t];
                        # later layers read the previous layer's activations.
                        # The padded tail of the first layer's width must be
                        # skipped outright: its weight is zero, but reading past
                        # x_norm is out of bounds, and 0 * NaN is NaN, which
                        # would poison an otherwise correct sum.
                        if layer_index == 0:
                            if k < head_dim:
                                total += weights[layer_index, hd, j, k] * h[b, hd, k]
                            elif k < 2 * head_dim:
                                total += (
                                    weights[layer_index, hd, j, k] * x_norm[b, hd, k - head_dim]
                                )
                        else:
                            total += weights[layer_index, hd, j, k] * scratch[layer_index - 1, b, k]
                    scratch[layer_index, b, j] = total
                # SiLU is x * sigmoid(x), not ReLU: it stays negative for x < 0
                # and tends to 0 from below, so clamping would change the result.
                if layer_index < depth - 1:
                    for j in range(width):
                        s = scratch[layer_index, b, j]
                        scratch[layer_index, b, j] = s / (1.0 + ti.exp(-s))
            for j in range(head_dim):
                out[b, hd, j] = scratch[depth - 1, b, j]

    return norm_kernel, step_kernel


def rnn_step(
    x: torch.Tensor,
    h_prev: torch.Tensor,
    rnn_layer: nn.Module,
    *args: object,
    **kwargs: object,
) -> torch.Tensor:
    """Advance one token through a multi-head MLP-RNN layer on the Taichi backend.

    Args:
        x: Input token, [B, D].
        h_prev: Previous per-head state, [B, H, D_head].
        rnn_layer: A ``MultiHeadRNNLayer`` whose heads are ``Linear`` layers
            with ``SiLU`` between them, of any depth and any width.

    Returns:
        The updated per-head state, [B, H, D_head].

    Raises:
        TypeError: the module is not a multi-head RNN layer.
        ValueError: the head topology is not an alternating Linear/SiLU stack,
            or an input has the wrong shape.
        RuntimeError: the layer is in training mode, where the kernel would
            silently drop the reference's dropout.
    """
    ensure_initialized()
    # Validate the module before its mode: an unsupported module should report
    # that, not the inference-only rule that only applies once it is supported.
    topology = _head_topology(rnn_layer)
    num_heads = int(rnn_layer.num_heads)  # type: ignore[attr-defined]
    head_dim = int(rnn_layer.head_dim)  # type: ignore[attr-defined]
    hidden_dim = int(rnn_layer.hidden_dim)  # type: ignore[attr-defined]

    if rnn_layer.training:
        raise RuntimeError(
            "the Taichi rnn_step kernel is inference-only and does not apply "
            "dropout; call it on a layer in eval() so it matches the reference"
        )

    first_in, last_out = topology[0][0], topology[-1][1]
    if first_in != head_dim * 2:
        raise ValueError(
            f"the head consumes {first_in} inputs, expected 2*head_dim={2 * head_dim} "
            "for the [h_prev, x_t] concatenation"
        )
    if last_out != head_dim:
        raise ValueError(f"the head produces {last_out} outputs, expected head_dim={head_dim}")

    if x.dim() != 2 or x.shape[1] != hidden_dim:
        raise ValueError(f"x must be [batch, {hidden_dim}], got {tuple(x.shape)}")
    batch = int(x.shape[0])
    if h_prev.dim() == 2:
        h_prev = h_prev.view(batch, num_heads, head_dim)
    if tuple(h_prev.shape[1:]) != (num_heads, head_dim):
        raise ValueError(
            f"h_prev must be [batch, {num_heads}, {head_dim}], got {tuple(h_prev.shape)}"
        )

    weights, biases = _head_weights(rnn_layer, topology)
    norm_w = rnn_layer.norm.weight.detach()  # type: ignore[attr-defined]
    eps = float(rnn_layer.norm.eps)  # type: ignore[attr-defined]

    depth = len(topology)
    # Padded width: the kernel runs the same width-wide loop for every layer, so
    # it must cover the widest layer and the concatenated first-layer input.
    width = max([2 * head_dim, *(max(in_d, out_d) for in_d, out_d, _ in topology)])
    has_bias = any(b for _, _, b in topology)

    key = (num_heads, head_dim, hidden_dim, depth, width, has_bias, x.dtype, str(x.device))
    kernels = _KERNEL_CACHE.get(key)
    if kernels is None:
        kernels = _build_kernel(num_heads, head_dim, hidden_dim, depth, width, has_bias)
        _KERNEL_CACHE[key] = kernels
    norm_kernel, step_kernel = kernels

    # The kernel reads weights as [depth, num_heads, out, in]; each layer's
    # [H, out, in] slice is copied into the top-left corner of a zero-padded
    # block, so the padded columns multiply by zero and contribute nothing.
    packed_weights = torch.zeros(depth, num_heads, width, width, dtype=x.dtype, device=x.device)
    packed_biases = torch.zeros(depth, num_heads, width, dtype=x.dtype, device=x.device)
    for layer_index, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
        out_d, in_d = weight.shape[-2], weight.shape[-1]
        packed_weights[layer_index, :, :out_d, :in_d] = weight
        if bias is not None:
            packed_biases[layer_index, :, :out_d] = bias

    x_norm = torch.empty(batch, num_heads, head_dim, dtype=x.dtype, device=x.device)
    scratch = torch.zeros(depth, batch, width, dtype=x.dtype, device=x.device)
    out = torch.empty(batch, num_heads, head_dim, dtype=x.dtype, device=x.device)
    norm_kernel(
        x.contiguous(),
        norm_w.contiguous(),
        eps,
        x_norm,
    )
    step_kernel(
        x_norm,
        h_prev.contiguous(),
        packed_weights,
        packed_biases,
        scratch,
        out,
    )
    return out

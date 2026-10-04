"""Tests for the HOPE recurrent cell (arXiv:2512.24695 §8).

Covers the memory math in isolation, the layer's forward/step contract, the
registered models, and the parallel-training path that HOPE takes through
:class:`ParallelRNNTrainer`.
"""

from __future__ import annotations

import math
from typing import cast

import pytest
import torch
from torch import nn

from nonlinearrnnscanbeparallel.models.base import RNNState
from nonlinearrnnscanbeparallel.models.hope import (
    HOPE,
    HOPE_MEMORIES,
    HOPE_OBJECTIVES,
    HOPELayer,
)
from nonlinearrnnscanbeparallel.models.hope_memory import (
    advance,
    l2_gradient,
    logit,
    retrieve,
)
from nonlinearrnnscanbeparallel.models.parallel_wrapper import (
    ParallelRNNTrainer,
    _layer_memory_shape,
    _target_state_rms,
)
from nonlinearrnnscanbeparallel.models.registry import get_model
from nonlinearrnnscanbeparallel.models.translator import TranslatorLayer, TranslatorStack

CPU = torch.device("cpu")


def _memories(batch: int = 2, heads: int = 3, n: int = 5) -> torch.Tensor:
    """A well-conditioned stack of memories, [B, H, N, N]."""
    torch.manual_seed(0)
    return torch.randn(batch, heads, n, n) * 0.5


# --------------------------------------------------------------------------
# Memory math
# --------------------------------------------------------------------------


def test_when_retrieving_then_result_equals_matrix_vector_product() -> None:
    """Retrieval is the plain linear map M(z) = M z."""
    m = _memories()
    z = torch.randn(2, 3, 5)

    torch.testing.assert_close(retrieve(m, z), torch.einsum("bhkv,bhk->bhv", m, z))


def test_when_l2_gradient_then_matches_autograd() -> None:
    """The hand-written L2 gradient equals autograd on ||M k - v||^2 / 2."""
    m = _memories()
    k = torch.randn(2, 3, 5)
    v = torch.randn(2, 3, 5)

    analytic = l2_gradient(m, k, v)

    m_ref = m.clone().requires_grad_(True)
    loss = 0.5 * (torch.einsum("bhkv,bhk->bhv", m_ref, k) - v).pow(2).sum()
    loss.backward()
    torch.testing.assert_close(analytic, m_ref.grad)


def test_when_eta_is_zero_then_update_is_pure_alpha_decay() -> None:
    """With eta = 0 the gradient term vanishes and the step is alpha I."""
    m = _memories()
    k = torch.randn(2, 3, 5)
    v = torch.randn(2, 3, 5)
    zero = torch.zeros(2, 3, 1, 1)

    kept = advance(m, k, v, zero, torch.ones_like(zero), "l2")
    shrunk = advance(m, k, v, zero, torch.full_like(zero, 0.5), "l2")

    torch.testing.assert_close(kept, m)
    torch.testing.assert_close(shrunk, 0.5 * m)


def test_when_alpha_halves_then_retained_memory_halves() -> None:
    """alpha multiplies the retained memory, for either objective."""
    m = _memories()
    k = torch.randn(2, 3, 5)
    v = torch.randn(2, 3, 5)
    one = torch.ones(2, 3, 1, 1)

    full = advance(m, k, v, torch.zeros(2, 3, 1, 1), one, "dot")
    half = advance(m, k, v, torch.zeros(2, 3, 1, 1), 0.5 * one, "dot")

    torch.testing.assert_close(half, 0.5 * full)


def test_when_objectives_differ_then_l2_and_dot_disagree() -> None:
    """The two paper objectives are genuinely different update rules."""
    m = _memories()
    k = torch.randn(2, 3, 5)
    v = torch.randn(2, 3, 5)
    eta = torch.full((2, 3, 1, 1), 0.1)
    alpha = torch.full((2, 3, 1, 1), 0.9)

    l2 = advance(m, k, v, eta, alpha, "l2")
    dot = advance(m, k, v, eta, alpha, "dot")

    assert not torch.allclose(l2, dot)


@pytest.mark.parametrize("objective", HOPE_OBJECTIVES)
def test_when_objective_is_valid_then_advance_leaves_memory_finite(
    objective: str,
) -> None:
    """A single valid step keeps every memory finite for both objectives."""
    m = _memories()
    k = torch.nn.functional.normalize(torch.randn(2, 3, 5), dim=-1)
    v = torch.randn(2, 3, 5)
    eta = torch.full((2, 3, 1, 1), 0.1)
    alpha = torch.full((2, 3, 1, 1), 0.99)

    updated = advance(m, k, v, eta, alpha, objective)

    assert updated.shape == m.shape
    assert torch.isfinite(updated).all()


def test_when_objective_is_unknown_then_advance_raises() -> None:
    """An unsupported objective is a programming error, not a silent fallback."""
    m = _memories()
    k = torch.randn(2, 3, 5)
    v = torch.randn(2, 3, 5)
    eta = torch.full((2, 3, 1, 1), 0.1)
    alpha = torch.full((2, 3, 1, 1), 0.9)

    with pytest.raises(ValueError, match="Unknown HOPE objective"):
        advance(m, k, v, eta, alpha, "huber")


@pytest.mark.parametrize("p", [0.99, 0.5, 0.01, 0.75])
def test_when_inverting_sigmoid_then_recovers_input(p: float) -> None:
    """logit inverts sigmoid away from the open-interval singularities."""
    assert torch.sigmoid(torch.tensor(logit(p))).item() == pytest.approx(p, abs=1e-6)


# --------------------------------------------------------------------------
# Layer protocol
# --------------------------------------------------------------------------


def test_when_initializing_state_then_shape_is_heads_by_five_matrices() -> None:
    """init_state reports the [B, H, 5, N, N] layout the wrapper reshapes into."""
    layer = HOPELayer(hidden_dim=32, num_heads=4, mem_dim=8)

    state = layer.init_state(2, CPU)

    assert state.hidden.shape == (2, 4, len(HOPE_MEMORIES), 8, 8)
    assert layer.stacked_state_shape() == (4, len(HOPE_MEMORIES), 8, 8)


def test_when_initializing_state_then_memories_are_seeded_not_zero() -> None:
    """A zero M_v would zero the regression target and kill every gradient."""
    layer = HOPELayer(hidden_dim=32, num_heads=4, mem_dim=8)

    state = layer.init_state(2, CPU)

    assert state.hidden.abs().sum() > 0


def test_when_forwarding_then_output_shape_and_state_are_consistent() -> None:
    """forward returns one output per token and the advanced state."""
    torch.manual_seed(0)
    layer = HOPELayer(hidden_dim=32, num_heads=4, mem_dim=8)
    x = torch.randn(2, 12, 32)

    out, state = layer(x)

    assert out.shape == x.shape
    assert state.hidden.shape == (2, 4, len(HOPE_MEMORIES), 8, 8)


@pytest.mark.parametrize(
    ("levels", "period", "taps"),
    [(1, 1, 1), (1, 1, 4), (2, 2, 3), (3, 1, 4)],
    ids=["single-tap", "one-level-multi-tap", "two-levels", "three-levels"],
)
def test_when_stepping_then_matches_full_forward(levels: int, period: int, taps: int) -> None:
    """Token-by-token decoding runs the same recurrence as full-sequence forward.

    The CMS reads history at several lags, so this only holds because that
    history is carried in the recurrent state rather than recomputed from the
    sequence.

    The recurrence is checked exactly with ``q`` held fixed, because ``forward``
    convolves the whole sequence at once while ``step`` convolves one token at
    a time. Those two sum the same causal window in a different order, so
    float32 makes them differ by ~3e-7, and Eq. 95's self-referential target
    amplifies that across the sequence. The end-to-end outputs are therefore
    only asserted to a loose tolerance; pinning them tightly would pin the
    conv's reduction order rather than the recurrence.
    """
    torch.manual_seed(0)
    layer = HOPELayer(
        hidden_dim=32,
        num_heads=4,
        mem_dim=8,
        num_cms_levels=levels,
        cms_period=period,
        cms_taps=taps,
        dropout=0.0,
    )
    layer.eval()
    x = torch.randn(2, 20, 32)
    cpu = torch.device("cpu")

    with torch.no_grad():
        state = None
        stepped = []
        for t in range(x.shape[1]):
            out_t, state = layer.step(x[:, t], state)
            stepped.append(out_t)
        full, full_state = layer(x, chunk_size=None)

        # Eq. 90's frozen-evaluation form, driven directly with the same q
        # forward computes. With a single chunk the frozen memory and the
        # running memory coincide, so this must reproduce forward exactly.
        init = layer.init_state(2, cpu)
        init_mem = layer._mem_from_state(init)
        q, _ = layer._q(x, cast(dict, init.extra)["q"])
        mem = list(init_mem)
        cms = cast(dict, init.extra)["cms"]
        outputs = []
        for t in range(x.shape[1]):
            # One chunk: the frozen memory is the memory at t=0 throughout.
            mem, o_t = layer._recurrence(mem, x[:, t], q[:, t], init_mem)
            out_t, cms = layer._project_token(o_t, cms)
            outputs.append(out_t + x[:, t])

    torch.testing.assert_close(torch.stack(mem, dim=2), full_state.hidden)
    torch.testing.assert_close(torch.stack(outputs, dim=1), full)
    # The two q paths differ only by float32 rounding of the same window.
    torch.testing.assert_close(torch.stack(stepped, dim=1), full, atol=2e-1, rtol=2e-1)


def test_when_cms_reads_history_then_buffer_matches_required_lag() -> None:
    """The carried CMS history is exactly as wide as the slowest tap."""
    layer = HOPELayer(
        hidden_dim=32, num_heads=4, mem_dim=8, num_cms_levels=2, cms_period=2, cms_taps=3
    )

    _, state = layer(torch.randn(2, 10, 32))

    lag_max = layer.cms_period * 2 ** (layer.num_cms_levels - 1) * (layer.cms_taps - 1)
    assert state.extra is not None
    assert state.extra["cms"].shape == (2, lag_max, 4 * 8)


def test_when_backpropagating_then_all_memory_gradients_are_finite_and_live() -> None:
    """Every parameter gets a usable gradient under the chunked training form.

    The inner memories reach the output only through Eq. 90's chunk-start
    evaluation point: ``o_t`` reads ``M_mem`` alone, and a perturbed inner
    memory changes the sequential output by exactly zero.  So the sequential
    form leaves them inert by construction, and gradient liveness has to be
    asserted on the chunked path that training actually runs.
    """
    torch.manual_seed(0)
    layer = HOPELayer(hidden_dim=32, num_heads=4, mem_dim=8)
    x = torch.randn(2, 12, 32)

    out, _ = layer(x, chunk_size=4)
    out.sum().backward()

    grads = {n: p.grad for n, p in layer.named_parameters() if p.requires_grad}
    assert grads, "layer exposed no trainable parameters"
    for name, grad in grads.items():
        assert grad is not None, f"{name} received no gradient"
        assert torch.isfinite(grad).all(), f"{name} gradient is not finite"
        assert grad.abs().sum() > 0, f"{name} gradient is identically zero"


def test_when_sequential_then_inner_memories_do_not_reach_the_output() -> None:
    """Eq. 94 reads M_mem alone, so the inner memories are inert sequentially.

    This is the reason Eq. 90 exists: the chunk-start memory is what makes the
    inner memories observable, and it is why chunkwise training is not merely
    an optimization of the sequential form but a different model.
    """
    torch.manual_seed(0)
    layer = HOPELayer(hidden_dim=32, num_heads=4, mem_dim=8)
    x = torch.randn(2, 8, 32)
    init = layer.init_state(2, torch.device("cpu"))

    perturbed = init.hidden.clone()
    perturbed[:, :, 1:] += 0.5
    shifted = RNNState(hidden=perturbed, extra=dict(init.extra or {}))

    sequential, _ = layer(x, init, chunk_size=None)
    shifted_out, _ = layer(x, shifted, chunk_size=None)
    torch.testing.assert_close(sequential, shifted_out)

    chunked, _ = layer(x, init, chunk_size=4)
    shifted_chunked, _ = layer(x, shifted, chunk_size=4)
    assert (chunked - shifted_chunked).abs().max() > 0, "Eq. 90 must expose the inner memories"


def test_when_cms_taps_is_zero_then_raises() -> None:
    """cms_taps must be positive; zero would silently disable the CMS."""
    with pytest.raises(ValueError, match="cms_taps"):
        HOPELayer(hidden_dim=32, num_heads=4, mem_dim=8, cms_taps=0)


def test_when_mem_dim_mismatches_head_width_then_raises_with_the_expected_value() -> None:
    """Eq. 94 needs mem_dim == head width; the error must say what to pass."""
    with pytest.raises(ValueError, match="expected mem_dim=7"):
        HOPELayer(hidden_dim=30, num_heads=4, mem_dim=8)


def test_when_repeated_steps_then_state_does_not_mutate_caller_input() -> None:
    """Decoding must not write back into the tensor the caller passed in."""
    torch.manual_seed(0)
    layer = HOPELayer(hidden_dim=32, num_heads=4, mem_dim=8)
    layer.eval()
    x = torch.randn(2, 10, 32)
    original = x.clone()

    with torch.no_grad():
        state = None
        for t in range(x.shape[1]):
            _, state = layer.step(x[:, t], state)

    torch.testing.assert_close(x, original)


# --------------------------------------------------------------------------
# Registered models
# --------------------------------------------------------------------------


def test_when_registered_then_hope_is_retrievable_from_the_registry() -> None:
    """`hope` resolves to the HOPE classifier model, not a stale alias."""
    import nonlinearrnnscanbeparallel.models  # noqa: F401  (registry side effect)

    model = get_model(
        "hope", input_dim=32, hidden_dim=32, num_layers=1, num_heads=4, num_classes=3, mem_dim=8
    )

    assert isinstance(model, HOPE)
    assert model.mixer_type == "hope"


def test_when_building_hope_model_then_forward_returns_class_logits() -> None:
    """The standalone HOPE model maps features to per-class logits."""
    torch.manual_seed(0)
    model = get_model(
        "hope",
        input_dim=32,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        num_classes=5,
        mem_dim=8,
        dropout=0.0,
    )

    logits, _ = model(torch.randn(2, 10, 32))

    assert logits.shape == (2, 10, 5)


def test_when_nano_rnn_uses_hope_mixer_then_matrices_are_in_the_state() -> None:
    """NanoRNN's HOPE mixer initializes each layer from its own memories."""
    import nonlinearrnnscanbeparallel.models  # noqa: F401

    model = get_model(
        "nano_rnn",
        input_dim=0,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        vocab_size=17,
        mixer_type="hope",
        mem_dim=8,
    )

    state = model.init_state(2, CPU)

    assert len(state) == 2
    for layer_state in state:
        assert layer_state.hidden.shape == (2, 4, len(HOPE_MEMORIES), 8, 8)


def test_when_hope_mixer_omits_mem_dim_then_it_follows_the_head_width() -> None:
    """NanoRNN derives HOPE's memory width so callers need not restate it.

    Without this, `mixer_type="hope"` fails deep inside einsum with a
    broadcast error whenever the caller forgets to pass mem_dim.
    """
    torch.manual_seed(0)
    model = get_model(
        "nano_rnn",
        input_dim=0,
        head_dim=8,
        num_layers=1,
        num_heads=4,
        vocab_size=17,
        mixer_type="hope",
    )

    state = model.init_state(2, CPU)

    assert model.mem_dim == 8
    assert state[0].hidden.shape == (2, 4, len(HOPE_MEMORIES), 8, 8)


def test_when_nano_rnn_uses_hope_mixer_then_generates_token_logits() -> None:
    """A HOPE NanoRNN language model consumes token IDs end to end."""
    torch.manual_seed(0)
    model = get_model(
        "nano_rnn",
        input_dim=0,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        vocab_size=17,
        mixer_type="hope",
        mem_dim=8,
    )
    model.eval()

    with torch.no_grad():
        logits, _ = model(torch.randint(0, 17, (2, 12)))

    assert logits.shape == (2, 12, 17)


# --------------------------------------------------------------------------
# Parallel training
# --------------------------------------------------------------------------


def _hope_wrapper(
    chunk_size: int = 8,
    model_kwargs: dict[str, object] | None = None,
    target_chunk_size: int | None = None,
) -> ParallelRNNTrainer:
    """A minimal HOPE language model wrapped for chunkwise parallel training.

    ``model_kwargs`` configure the target; ``target_chunk_size`` configures
    Eq. 90's inner chunk in the wrapper.
    """
    import nonlinearrnnscanbeparallel.models  # noqa: F401

    params: dict[str, object] = {
        "input_dim": 0,
        "hidden_dim": 32,
        "num_layers": 2,
        "num_heads": 4,
        "vocab_size": 17,
        "mixer_type": "hope",
        "mem_dim": 8,
    }
    params.update(model_kwargs or {})
    target = get_model("nano_rnn", **params)
    return ParallelRNNTrainer(
        target, chunk_size=chunk_size, scaffold_dim=16, target_chunk_size=target_chunk_size
    )


def test_when_inferring_target_type_then_hope_is_a_matrix_target() -> None:
    """HOPE must route to the matrix-state branch, not the vector one."""
    wrapper = _hope_wrapper()

    assert wrapper.target_type == "hope"
    assert (
        "hope"
        in __import__(
            "nonlinearrnnscanbeparallel.models.parallel_wrapper", fromlist=["MATRIX_TARGET_TYPES"]
        ).MATRIX_TARGET_TYPES
    )


def test_when_reading_memory_shape_then_includes_the_memory_count() -> None:
    """The wrapper needs (num_memories, N, N) to size the translator."""
    import nonlinearrnnscanbeparallel.models  # noqa: F401

    target = get_model(
        "nano_rnn",
        input_dim=0,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        vocab_size=17,
        mixer_type="hope",
        mem_dim=8,
    )

    assert _layer_memory_shape(target) == (len(HOPE_MEMORIES), 8, 8)


def test_when_constructing_wrapper_then_translator_emits_the_full_state() -> None:
    """The translated boundary reshapes directly into the layer state."""
    wrapper = _hope_wrapper()
    translator = cast(TranslatorStack, wrapper.translators[0])
    last = cast(TranslatorLayer, translator.layers[-1])

    out = translator(torch.randn(2, 3, 16))

    assert last.num_memories == len(HOPE_MEMORIES)
    assert out.shape == (2, 3, 4, len(HOPE_MEMORIES), 8, 8)


def test_when_target_state_is_tiny_then_boundary_scale_matches_it() -> None:
    """The translator's RMSNorm output is rescaled into the target's range."""
    wrapper = _hope_wrapper()

    with torch.no_grad():
        boundary = wrapper.translators[0](torch.randn(2, 3, 16))
    target_rms = _target_state_rms(wrapper.target)

    assert 0 < target_rms < 1
    assert float(boundary.pow(2).mean().sqrt()) == pytest.approx(target_rms, rel=0.6)


def test_when_training_in_parallel_then_logits_cover_every_token() -> None:
    """Chunked parallel training returns one logit per timestep."""
    torch.manual_seed(0)
    wrapper = _hope_wrapper(chunk_size=8)
    wrapper.train()

    logits = wrapper(torch.randint(0, 17, (2, 32)))

    assert logits.shape == (2, 32, 17)


def test_when_training_in_parallel_then_every_parameter_gets_a_gradient() -> None:
    """Target, scaffold, and translator all receive live gradients."""
    torch.manual_seed(0)
    wrapper = _hope_wrapper(chunk_size=8)
    wrapper.train()

    wrapper(torch.randint(0, 17, (2, 32))).sum().backward()

    groups = {"target": 0, "scaffold": 0, "translator": 0}
    for name, param in wrapper.named_parameters():
        for group in groups:
            if group in name and param.grad is not None and param.grad.abs().sum() > 0:
                groups[group] += 1
                break
    assert all(count > 0 for count in groups.values()), groups


def test_when_evaluating_then_sequential_inference_is_finite() -> None:
    """Eval runs the target sequentially, bypassing scaffold and translator."""
    torch.manual_seed(0)
    wrapper = _hope_wrapper(chunk_size=8)
    wrapper.eval()

    with torch.no_grad():
        logits = wrapper(torch.randint(0, 17, (2, 32)))

    assert logits.shape == (2, 32, 17)
    assert torch.isfinite(logits).all()


def test_when_optimizing_then_loss_decreases() -> None:
    """A short optimization run makes real progress on HOPE boundaries."""
    torch.manual_seed(0)
    wrapper = _hope_wrapper(chunk_size=8)
    wrapper.train()
    x = torch.randint(0, 17, (2, 32))
    y = torch.randint(0, 17, (2, 32))
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-3)

    losses = []
    for _ in range(12):
        logits = wrapper(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, 17), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))

    assert all(loss == loss for loss in losses), "loss became NaN"
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"


def test_when_chunk_size_exceeds_sequence_then_falls_back_to_sequential() -> None:
    """A single chunk still trains, using the layer's own initial state."""
    torch.manual_seed(0)
    wrapper = _hope_wrapper(chunk_size=64)
    wrapper.train()

    with pytest.warns(UserWarning, match="single-chunk"):
        logits = wrapper(torch.randint(0, 17, (2, 16)))

    assert logits.shape == (2, 16, 17)


def test_when_zero_memory_then_v_memory_gets_no_gradient() -> None:
    """The defect the chunk-0 init guards against.

    A zero M_v retrieves v_t = 0, so the self-referential chain receives no
    gradient at all. Chunk 0 must therefore start from the layer's learned
    memories, never from zeros.
    """
    torch.manual_seed(0)
    layer = HOPELayer(hidden_dim=32, num_heads=2, mem_dim=16)
    x = torch.randn(2, 8, 32)

    layer(x, RNNState(hidden=torch.zeros(2, 2, 5, 16, 16)))[0].pow(2).mean().backward()

    assert layer.v_memory.grad is None


def test_when_parallel_chunked_then_every_layer_receives_memory_gradient() -> None:
    """Every layer's memories must train, not just layer 0.

    Layers after the first start from a placeholder initial state. Zeroing that
    placeholder silently drops the whole upper stack from optimization, so each
    layer must be seeded with its own learned memories.

    ``target_chunk_size`` must be strictly below the scaffold chunk: HOPE's
    inner memories only reach the output through Eq. 90's chunk-start
    evaluation point, so with one evaluation point per chunk they receive
    exactly zero gradient.
    """
    torch.manual_seed(0)
    wrapper = _hope_wrapper(chunk_size=8, target_chunk_size=4)
    wrapper.train()

    logits = wrapper(torch.randint(0, 17, (2, 32)))
    logits.pow(2).mean().backward()

    for index in range(wrapper.num_layers):
        rnn_layer, _ = wrapper._get_layer_pair(index)
        memory_params = {n: p for n, p in rnn_layer.named_parameters() if n.endswith("_memory")}
        dead = [
            n
            for n, p in memory_params.items()
            if p.grad is None or float(p.grad.abs().sum()) == 0.0
        ]
        assert len(memory_params) == len(HOPE_MEMORIES)
        assert not dead, f"layer {index} memories got no gradient: {dead}"


def test_when_target_chunk_size_is_omitted_then_inner_memories_stay_dead() -> None:
    """Defaults must not silently train only M_mem.

    With one evaluation point per chunk, Eq. 90 degenerates to the sequential
    form, where the inner memories provably do not affect the output. Pinning
    that makes the requirement for ``target_chunk_size`` visible.
    """
    torch.manual_seed(0)
    wrapper = _hope_wrapper(chunk_size=8)
    wrapper.train()

    wrapper(torch.randint(0, 17, (2, 32))).pow(2).mean().backward()
    layer, _ = wrapper._get_layer_pair(0)
    inner = {
        n: (0.0 if p.grad is None else float(p.grad.abs().sum()))
        for n, p in layer.named_parameters()
        if n.endswith("_memory") and not n.startswith("mem_")
    }

    assert inner, "no inner memories found to check"
    assert all(value == 0.0 for value in inner.values())


def test_when_wrapper_is_constructed_then_upper_layers_are_seeded_not_zeroed() -> None:
    """The placeholder state for an unvisited layer carries the learned init."""
    torch.manual_seed(0)
    wrapper = _hope_wrapper()
    rnn_layer, _ = wrapper._get_layer_pair(1)

    placeholder = wrapper._initial_state_for(1, 2, CPU, torch.float32)
    expected = rnn_layer.init_state(2, CPU).hidden  # type: ignore[attr-defined]

    assert placeholder.hidden.shape == expected.shape
    assert not torch.allclose(placeholder.hidden, torch.zeros_like(placeholder.hidden))
    torch.testing.assert_close(placeholder.hidden, expected)


def test_when_training_shipped_geometry_then_loss_stays_finite() -> None:
    """The two-layer wrapper trains without diverging at the shipped settings.

    Small-vocabulary probes (vocab 97) do diverge within a handful of steps at
    any clip threshold, so this pins the geometry the configs actually use.
    """
    torch.manual_seed(0)
    vocab = 512
    model = get_model(
        "nano_rnn",
        input_dim=0,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        vocab_size=vocab,
        mixer_type="hope",
        mem_dim=16,
    )
    wrapper = ParallelRNNTrainer(model, chunk_size=16, scaffold_dim=64)
    wrapper.train()
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=3e-4)
    torch.manual_seed(1)
    ids = torch.randint(0, vocab, (2, 64))
    targets = torch.randint(0, vocab, (2, 64))

    losses = []
    for _ in range(6):
        logits = wrapper(ids)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
        optimizer.step()

    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] < losses[0]

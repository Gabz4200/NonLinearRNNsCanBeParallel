"""HOPE: self-modifying Titans with Continuum Memory (arXiv:2512.24695 §8).

HOPE is a recurrent cell whose state is a set of learned associative memories
that adapt *in-context* through their own gradient steps.  Per token ``t``:

    o_t     = M_mem,t-1(q_t)                              retrieval (Eq. 94)
    k_t     = M_k,t-1(x_t)        v_t     = M_v,t-1(x_t)
    eta_t   = M_eta,t-1(x_t)      alpha_t = M_alpha,t-1(x_t)
    v_hat_s = M_s,t-1(v_t)        for s in {k, v, eta, alpha}   (Eq. 95)

    M_mem,t = upd(M_mem,t-1; k_t, v_t)
    M_s,t   = upd(M_s,t-1;   k_t, v_hat_s)                          (Eq. 96)

    y_t = MLP^{f_k}( ... MLP^{f_1}( o_t ) )                    CMS (Eq. 97)

Only ``q`` is a fixed projection (paper §8.1: "q_t = x_t W_q is the only
non-adaptive projection"); every other signal comes from a memory, which is
what makes the cell *self-referential* rather than a Titans block.

State geometry.  Each memory is one square ``N x N`` matrix per head, giving a
layer state of ``[B, H, 5, N, N]`` (heads first, then the memory axis).  The
square shape is forced by Eq. 93, which right-multiplies the memory by the
key-space factor ``alpha I - eta k k^T`` (see :mod:`.hope_memory`).  This is a
matrix-state target, matching the geometry :class:`ParallelRNNTrainer` already
handles for M2RNN.

The recurrence is *sequential*: the error term ``M k`` depends on the current
memory, so the dual form is not chunk-scannable within a chunk.  Chunkwise
parallel training therefore uses this repository's scaffold/translator
decomposition, which supplies each chunk's initial memories from a parallel
scanner.

Eq. 90 and why the inner chunk matters.  The paper's chunkwise form evaluates
every signal and gradient at the chunk-start memory,
``M_{□,C x ceil(t/C)}``, rather than the running one.  That is not merely a
parallelism device here, it is load-bearing twice over:

* *Stability.*  Because Eq. 95's target is self-generated, the four inner
  memories form a positive feedback loop: a larger ``M_s`` gives a larger
  ``v_hat = M_s v_t``, which enlarges the gradient ``(M_s k - v_hat) k^T`` and
  so enlarges ``M_s`` again.  The decay term is contractive (``alpha < 1``), so
  only the target term runs away, and the loop is superlinear.  Evaluated on the
  running memory it diverges to NaN within ~13 optimizer steps at two layers,
  growing geometrically (1e-1 -> 1e+0 -> 1e+2 -> 1e+8 -> 1e+34).  Freezing the
  evaluation point at the chunk start removes the self-reference and leaves the
  update intact.
* *Trainability.*  ``o_t`` reads ``M_mem`` alone, so in the plain sequential
  form a perturbed inner memory changes the output by *exactly* zero: the inner
  memories are structurally inert.  They become observable only through Eq. 90's
  chunk-start evaluation point.  Consequently ``parallel.target_chunk_size``
  must be strictly below the scaffold ``chunk_size``; equal to it, Eq. 90
  degenerates to the sequential form and ``k``/``v``/``eta``/``alpha`` receive
  exactly zero gradient.

:meth:`forward` takes ``chunk_size`` for that reason and defaults to the plain
sequential form, which is what :meth:`step` always runs, so decoding token by
token matches sequence evaluation.

Known approximation under chunkwise training: the translated boundary carries
the five memories but not the ``q`` conv cache or the Continuum Memory history,
since both are token history the scaffold never sees.  Both therefore restart at
each chunk boundary.  Measured at the layer level over 64 tokens with chunk 16,
this costs up to 0.75 of output deviation from a sequential run, and the CMS
history dominates the error (carrying it alone recovers all but 4e-7, while
carrying the ``q`` cache alone barely helps).  Carrying both across chunks was
implemented and measured, and reverted: the gain is invisible at the logits
level, because deviation from sequential is dominated by memory translation
either way, so the extra sequential dependency bought nothing observable.

Two float32 caveats, both properties of the recurrent feedback rather than
logic errors.  :meth:`forward` convolves ``q`` over the whole sequence while
:meth:`step` convolves one token at a time; both sum the same causal window in a
different order, so ``q`` differs by ~3e-7, which the self-referential target
amplifies over the sequence (to ~3e-2 after 20 tokens on a 32-wide layer).  For
the same reason a 1e-7 perturbation of ``q`` need not stay 1e-7.

Deviations from the paper, all deliberate and local:
  - Memories are linear matrices (see :mod:`.hope_memory` for why: Eqs. 92-93
    are derived only for that case).
  - q and k are L2-normalized and a depthwise causal conv (window 4) precedes
    the q projection, per paper §8.3.
  - ``eta`` is bounded by ``eta_scale * sigmoid(.)`` and ``alpha`` by
    ``sigmoid(. + alpha_logit)`` with ``alpha_logit`` initialized so retention
    starts at ``alpha_init``.  A plain sigmoid at zero would give ``alpha =
    0.5`` and halve the memory every token.
  - Continuum Memory levels are applied to the retrieved output, with level
    ``l`` read at lags ``period * 2^l`` and its multiples.  The paper's
    periodic in-loop *parameter* update (its Eq. 71 schedule) is deliberately
    not implemented: it would mutate parameters between autograd graph
    traversals and fight the outer optimizer.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from .base import RMSNorm, RNNState, RNNStateList, SequentialRNNModel
from .hope_memory import HOPE_MEMORIES, HOPE_OBJECTIVES, advance, logit, retrieve
from .m2rnn import DepthwiseConv1d
from .mlp_rnn import FeedForwardSublayer
from .registry import register_model

DEFAULT_HOPE_MEM_DIM = 64


class HOPELayer(nn.Module):
    """One HOPE layer: five self-modifying matrix memories plus a CMS tail.

    Layer state is a single ``[B, H, 5, N, N]`` tensor holding the five
    memories in :data:`HOPE_MEMORIES` order (heads first, then memories, so
    the layout matches M2RNN's ``[B, H, K, V]`` and the translator's matrix
    output), with the causal conv cache in ``extra`` under the ``"q"`` key.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        mem_dim: int = 64,
        conv_kernel: int = 4,
        dropout: float = 0.0,
        num_cms_levels: int = 1,
        cms_period: int = 1,
        cms_taps: int = 4,
        eta_scale: float = 0.1,
        alpha_init: float = 0.99,
        normalize_qk: bool = True,
        objective: str = "l2",
    ) -> None:
        """Args:
        hidden_dim: model width; each memory reads a head slice of it.
        num_heads: independent memory heads.
        mem_dim: per-head memory width ``N``; each memory is ``[H, N, N]``.
        conv_kernel: local causal conv window (paper §8.3 uses 4).
        dropout: output dropout.
        num_cms_levels: Continuum Memory System depth in Eq. 97; level ``l`` reads
            history at stride ``cms_period * 2^l``.
        cms_period: base CMS stride, in tokens.
        cms_taps: history taps each CMS level reads (0, stride, 2*stride, ...).
        eta_scale: upper bound on the internal learning rate.
        alpha_init: retention the alpha memory starts at.
        normalize_qk: L2-normalize q and k (paper §8.3).
        objective: "l2" (Eq. 93, used for the reported LM results) or "dot"
            (Eq. 92).
        """
        super().__init__()
        if objective not in HOPE_OBJECTIVES:
            raise ValueError(f"Unknown HOPE objective: {objective}")
        if num_cms_levels < 1:
            raise ValueError("num_cms_levels must be >= 1")
        if cms_period < 1:
            raise ValueError("cms_period must be >= 1")
        if cms_taps < 1:
            raise ValueError("cms_taps must be >= 1")
        if hidden_dim != num_heads * mem_dim:
            # Eq. 94 retrieves each signal with M_k x_heads, so the memory key
            # width must match the per-head input width. A mismatch surfaces
            # later as an opaque einsum broadcast error, so reject it here.
            raise ValueError(
                f"HOPE requires mem_dim == hidden_dim // num_heads "
                f"(got mem_dim={mem_dim}, hidden_dim={hidden_dim}, "
                f"num_heads={num_heads} -> expected mem_dim={hidden_dim // num_heads})"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mem_dim = mem_dim
        self.conv_kernel = conv_kernel
        self.num_cms_levels = num_cms_levels
        self.cms_period = cms_period
        self.eta_scale = eta_scale
        self.normalize_qk = normalize_qk
        self.objective = objective

        # q is the only non-adaptive projection (paper §8.1).
        self.W_q = nn.Linear(hidden_dim, num_heads * mem_dim, bias=False)
        self.q_conv = DepthwiseConv1d(num_heads * mem_dim, kernel_size=conv_kernel)

        self.alpha_logit = nn.Parameter(torch.full((num_heads,), logit(alpha_init)))
        self.eta_logit = nn.Parameter(torch.zeros(num_heads))

        # Each memory is a learned square matrix per head. They must NOT start
        # at zero: with M_v = 0 the retrieved v_t is 0, so M_mem's target is 0
        # and every gradient through the self-referential chain vanishes.
        for name in HOPE_MEMORIES:
            self.register_parameter(
                f"{name}_memory", nn.Parameter(torch.empty(num_heads, mem_dim, mem_dim))
            )
        self._init_memory_weights()

        cms_dim = num_heads * mem_dim
        self.cms_taps = cms_taps
        self.cms_layers = nn.ModuleList(
            [
                nn.Sequential(
                    RMSNorm(cms_dim * cms_taps),
                    nn.Linear(cms_dim * cms_taps, cms_dim, bias=False),
                    nn.SiLU(),
                    nn.Linear(cms_dim, cms_dim, bias=False),
                )
                for _ in range(num_cms_levels)
            ]
        )
        self.rms_scale = nn.Parameter(torch.ones(num_heads, mem_dim))
        self.output_proj = nn.Linear(cms_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _init_memory_weights(self) -> None:
        """Small-noise init, so every memory starts near a random projection.

        A zero or identity-dominant start either kills the self-referential
        gradient chain or freezes the output at the embedding.
        """
        for name in HOPE_MEMORIES:
            nn.init.normal_(getattr(self, f"{name}_memory"), mean=0.0, std=0.02)

    def stacked_state_shape(self) -> tuple[int, int, int, int]:
        """``(H, num_memories, N, N)``: the layer's full state geometry.

        The parallel wrapper and translator size themselves from this; a single
        memory is only ``(H, N, N)``, which is not the layer state.
        """
        return (self.num_heads, len(HOPE_MEMORIES), self.mem_dim, self.mem_dim)

    def conv_cache_dim(self) -> int:
        """Width of the convolved q projection, for sizing the conv cache."""
        return self.num_heads * self.mem_dim

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the model width into ``num_heads`` slices for memory reads.

        Falls back to broadcasting the whole vector to every head when
        ``hidden_dim`` is not divisible by ``num_heads``.
        """
        if self.hidden_dim % self.num_heads == 0:
            return x.view(x.shape[0], self.num_heads, self.hidden_dim // self.num_heads)
        return x.view(x.shape[0], 1, self.hidden_dim).expand(-1, self.num_heads, -1)

    def _q(
        self, x: torch.Tensor, conv_cache: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convolved, normalized q for [B, T, D]; returns [B, T, H, N] and cache."""
        conv, new_cache = self.q_conv(self.W_q(x), conv_cache)
        q = functional.silu(conv).view(x.shape[0], x.shape[1], self.num_heads, self.mem_dim)
        if self.normalize_qk:
            q = functional.normalize(q, dim=-1, eps=1e-6)
        return q, new_cache

    def _mem_from_state(self, state: RNNState) -> list[torch.Tensor]:
        """Split the stacked state into the five memory tensors."""
        return [state.hidden[:, :, i] for i in range(len(HOPE_MEMORIES))]

    def _resolve_state(self, state: RNNState | None, x: torch.Tensor) -> RNNState:
        """Return ``state``, seeding it from the learned memories when None."""
        if state is not None:
            return state
        return self.init_state(x.shape[0], x.device)

    def _signals(self, x_heads: torch.Tensor, mem: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Generate k, v, eta, alpha from their own memories (Eq. 94).

        Args:
            x_heads: [B, H, D/H] per-head slice of the input.
            mem: the five memory tensors, in :data:`HOPE_MEMORIES` order.

        Returns:
            k_t [B, H, N], v_t [B, H, N], eta_t [B, H, 1, 1], alpha_t [B, H, 1, 1].
        """
        k_t = retrieve(mem[1], x_heads)
        if self.normalize_qk:
            k_t = functional.normalize(k_t, dim=-1, eps=1e-6)
        v_t = retrieve(mem[2], x_heads)

        # eta and alpha are scalars per head, so they pool over the memory width.
        eta_raw = retrieve(mem[3], x_heads).mean(dim=-1, keepdim=True)
        alpha_raw = retrieve(mem[4], x_heads).mean(dim=-1, keepdim=True)
        eta_t = self.eta_scale * torch.sigmoid(eta_raw + self.eta_logit.view(1, -1, 1))
        alpha_t = torch.sigmoid(alpha_raw + self.alpha_logit.view(1, -1, 1))
        return k_t, v_t, eta_t.unsqueeze(-1), alpha_t.unsqueeze(-1)

    def _recurrence(
        self,
        mem: list[torch.Tensor],
        x_t: torch.Tensor,
        q_t: torch.Tensor,
        chunk_mem: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Advance all five memories one token; return new memories and ``o_t``.

        Args:
            mem: memories carried within the current chunk, i.e. the running
                state of Eq. 96's recurrent form.
            x_t: input at this token, [B, D_in].
            q_t: normalized query at this token, [B, H, N].
            chunk_mem: Eq. 90's ``M_{□,C x ceil(t/C)}``, the memory at the
                start of the enclosing chunk.  When given, the signals and the
                gradient are evaluated against this frozen memory while the
                returned state still carries ``mem`` forward, which is what
                makes the step parallelizable across a chunk.  None keeps the
                plain sequential form.

        Returns:
            The updated memories and the retrieved ``o_t``.
        """
        # Eq. 90 evaluates every signal and gradient at the chunk-start memory,
        # not the running one. Without this the inner memories run away: their
        # target is self-generated (Eq. 95), so a larger M_s gives a larger
        # v_hat = M_s v_t, which enlarges the gradient (M_s k - v_hat) k^T and
        # so enlarges M_s again. The loop is superlinear and diverges within a
        # couple of dozen tokens. Freezing the evaluation point removes the
        # self-reference while leaving the update itself intact.
        eval_mem = mem if chunk_mem is None else chunk_mem

        k_t, v_t, eta_t, alpha_t = self._signals(self._heads(x_t), eval_mem)

        # Retrieve before updating, so o_t reflects M_mem at t-1 (Eq. 94).
        o_t = retrieve(eval_mem[0], q_t)

        new_mem = [advance(eval_mem[0], k_t, v_t, eta_t, alpha_t, self.objective)]
        # Self-generated targets (Eq. 95): each inner memory writes toward its
        # own retrieval of v_t rather than toward v_t itself.
        for idx in range(1, len(HOPE_MEMORIES)):
            v_hat = retrieve(eval_mem[idx], v_t)
            new_mem.append(advance(eval_mem[idx], k_t, v_hat, eta_t, alpha_t, self.objective))
        return new_mem, o_t

    def _cms_lag_max(self) -> int:
        """Slowest lag any CMS level reads, i.e. the history buffer width."""
        slowest = self.cms_period * (2 ** max(self.num_cms_levels - 1, 0))
        return slowest * max(self.cms_taps - 1, 0)

    def _cms(
        self, o_t: torch.Tensor, cache: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the Continuum Memory System by one token (Eq. 97).

        Each level mixes the current retrieved output with its own history at
        an exponentially spaced stride, giving the layer a bank of timescales
        rather than a single-step nonlinearity.  Level ``l`` reads lags
        ``0, period*2^l, 2*period*2^l, ...``.

        Running this per token (with the history carried in the recurrent
        state) is what keeps :meth:`step` and :meth:`forward` identical: the
        taps cannot reach past the start of the sequence.

        Args:
            o_t: retrieved output at this token, [B, H * N].
            cache: [B, lag_max, H * N] of the previous ``lag_max`` outputs,
                with ``o_{t-1}`` at index 0. None at the start of a sequence.

        Returns:
            The CMS-refined output [B, H * N] and the updated buffer. The
            buffer is empty when the levels need no history.
        """
        lag_max = self._cms_lag_max()
        width = o_t.shape[-1]
        empty = o_t.new_zeros(o_t.shape[0], 0, width)
        buffer = cache if cache is not None and lag_max > 0 else empty
        y = o_t
        for level, layer in enumerate(self.cms_layers):
            stride = self.cms_period * (2**level)
            taps = [y]
            for tap in range(1, self.cms_taps):
                lag = stride * tap
                # Buffer index lag-1 is o_{t-lag}; anything past the buffer
                # start has no history yet and reads as zero.
                taps.append(
                    buffer[:, lag - 1]
                    if lag <= buffer.shape[1]
                    else o_t.new_zeros(o_t.shape[0], width)
                )
            y = y + layer(torch.cat(taps, dim=-1))
        if lag_max == 0:
            return y, empty
        # Keep exactly lag_max entries so index lag-1 is always o_{t-lag}.
        new_buffer = torch.cat([y.unsqueeze(1), buffer[:, : lag_max - 1]], dim=1)
        return y, new_buffer

    def _project_token(
        self, o_t: torch.Tensor, cms_cache: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CMS tail, per-head RMS scale, and output projection for one token."""
        width = self.num_heads * self.mem_dim
        y, cms = self._cms(o_t.reshape(o_t.shape[0], width), cms_cache)
        y = y.view(-1, self.num_heads, self.mem_dim) * self.rms_scale
        return self.dropout(self.output_proj(y.reshape(-1, width))), cms

    def forward(
        self, x: torch.Tensor, state: RNNState | None = None, chunk_size: int | None = None
    ) -> tuple[torch.Tensor, RNNState]:
        """Pass over [B, T, D_in]; returns (output [B, T, D], new_state).

        Args:
            x: input [B, T, D_in].
            state: incoming recurrent state, or None to seed from the learned
                memories.
            chunk_size: when set, the tokens per chunk of Eq. 90. Inside a chunk
                every signal and gradient is evaluated at the chunk-start memory
                while the returned state carries the running one forward, which
                is what makes the chunk parallelizable. None (the default) runs
                the plain sequential recurrence, where the running memory is
                always the evaluation point.
        """
        state = self._resolve_state(state, x)
        extra = state.extra or {}
        batch, t_len, _ = x.shape
        cache = extra.get("q")
        q, conv_cache = self._q(x, cache)
        mem = self._mem_from_state(state)
        # Normalize to a real tensor: the loop below always stores a tensor, and
        # an empty history is indistinguishable from a zero-filled one.
        cms_cache = extra.get("cms")
        if cms_cache is None:
            cms_cache = x.new_zeros(batch, self._cms_lag_max(), self.num_heads * self.mem_dim)

        outs: list[torch.Tensor] = []
        # chunk_size >= t_len means one chunk, i.e. the sequential form, so the
        # chunk-start memory and the running memory coincide.
        step_chunk = t_len if not chunk_size or chunk_size >= t_len else chunk_size
        for start in range(0, t_len, step_chunk):
            end = min(start + step_chunk, t_len)
            chunk_mem = mem
            for step in range(start, end):
                mem, o_t = self._recurrence(mem, x[:, step], q[:, step], chunk_mem)
                out_t, cms_cache = self._project_token(o_t, cms_cache)
                outs.append(out_t)

        out = torch.stack(outs, dim=1)
        # Each mem entry is [B, H, N, N]; dim=2 puts the 5 memories on axis 2
        # so the state is [B, H, 5, N, N], matching init_state.
        return out + x, RNNState(
            hidden=torch.stack(mem, dim=2),
            extra={"q": conv_cache, "cms": cms_cache},
        )

    def step(
        self, x_t: torch.Tensor, state: RNNState | None = None
    ) -> tuple[torch.Tensor, RNNState]:
        """Single-token step: x_t [B, D_in] -> (output [B, D], new_state).

        Always the sequential form, never Eq. 90's chunked one, so decoding
        token by token reproduces :meth:`forward` with ``chunk_size=None``.
        """
        state = self._resolve_state(state, x_t)
        cache = state.extra.get("q") if state.extra else None
        q, conv_cache = self._q(x_t.unsqueeze(1), cache)
        mem = self._mem_from_state(state)
        cms_cache = state.extra.get("cms") if state.extra else None

        mem, o_t = self._recurrence(mem, x_t, q[:, 0], None)
        out, cms_cache = self._project_token(o_t, cms_cache)
        return out + x_t, RNNState(
            hidden=torch.stack(mem, dim=2), extra={"q": conv_cache, "cms": cms_cache}
        )

    def init_state(self, batch_size: int, device: torch.device) -> RNNState:
        """Memories seeded from the learned matrices, plus a zero conv cache.

        The state is the batch-expanded learned memory, not zeros: a zero
        ``M_v`` yields ``v_t = 0``, which zeroes the regression target of
        ``M_mem`` and kills the gradient of the whole self-referential chain.

        Returns:
            [B, H, 5, N, N] memories (heads first, then the memory axis).
        """
        seeded = torch.stack([getattr(self, f"{name}_memory") for name in HOPE_MEMORIES], dim=1)
        return RNNState(
            hidden=seeded.unsqueeze(0).expand(batch_size, *seeded.shape).to(device),
            extra={
                "q": torch.zeros(
                    batch_size, self.conv_cache_dim(), self.conv_kernel - 1, device=device
                ),
                "cms": torch.zeros(
                    batch_size,
                    self._cms_lag_max(),
                    self.num_heads * self.mem_dim,
                    device=device,
                ),
            },
        )


@register_model("hope")
class HOPE(SequentialRNNModel):
    """HOPE model per arXiv:2512.24695 §8.3.

    ``emb_norm -> [HOPELayer, FeedForwardSublayer] x num_layers -> final_norm
    -> classifier``.  Declares ``mixer_type = "hope"`` so
    :class:`ParallelRNNTrainer` infers the right ``target_type``, and exposes
    the ``layers`` pair layout and ``init_state`` the wrapper expects.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        mem_dim: int = 64,
        conv_kernel: int = 4,
        num_cms_levels: int = 1,
        cms_period: int = 1,
        cms_taps: int = 4,
        eta_scale: float = 0.1,
        alpha_init: float = 0.99,
        normalize_qk: bool = True,
        objective: str = "l2",
        mlp_hidden_mult: int = 4,
        num_classes: int = 2,
        **kwargs: object,
    ) -> None:
        super().__init__(input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.mixer_type = "hope"
        self.mem_dim = mem_dim

        self.emb_norm = RMSNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        HOPELayer(
                            hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            mem_dim=mem_dim,
                            conv_kernel=conv_kernel,
                            dropout=dropout,
                            num_cms_levels=num_cms_levels,
                            cms_period=cms_period,
                            cms_taps=cms_taps,
                            eta_scale=eta_scale,
                            alpha_init=alpha_init,
                            normalize_qk=normalize_qk,
                            objective=objective,
                        ),
                        FeedForwardSublayer(hidden_dim, mlp_hidden_mult, dropout),
                    ]
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)

    def init_state(self, batch_size: int, device: torch.device) -> RNNStateList:
        """Per-layer states from each layer's own ``init_state``.

        Memories are seeded from the learned matrices, not zeros; see
        :meth:`HOPELayer.init_state` for why a zero start kills the
        self-referential gradient chain.
        """
        return RNNStateList(
            [layer_pair[0].init_state(batch_size, device) for layer_pair in self.layers]
        )


__all__ = ["HOPE", "HOPELayer"]

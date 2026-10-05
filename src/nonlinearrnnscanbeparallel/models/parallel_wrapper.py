"""Parallel RNN Training Wrapper.

Implements the chunkwise parallel training framework:
1. Per-layer scaffold scans the layer's input to produce boundary summaries
2. Translator maps scaffold summaries to target RNN boundary states
3. Target RNN runs in parallel across chunks from boundary states
4. Gradients are isolated: the target's own recurrence never crosses chunks,
   so target params see within-chunk BPTT only, while scaffold/translator
   learn solely via the boundary path (translator outputs -> chunk inits).

The wrapper exposes hidden states so custom output heads and losses can be attached.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as functional
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from .base import RNNState, RNNStateList
from .scaffold import ScaffoldStack
from .translator import _LEGACY_TARGET_TO_TRANSLATOR, TranslatorStack

# Targets whose recurrent state is a matrix per head rather than a vector, and
# which therefore need a matrix-shaped translator and the chunked matrix path.
MATRIX_TARGET_TYPES = ("m2rnn", "hope")


def _layer_memory_shape(target_rnn: nn.Module) -> tuple[int, ...]:
    """Per-head memory geometry of a matrix-state target, without the head axis.

    M2RNN reports ``(key_dim, value_dim)``; HOPE stacks five square memories on
    dim 1, so it reports ``(num_memories, N, N)``.  The translator emits
    ``num_heads * prod(shape)`` units per boundary, which reshapes straight into
    the layer state.

    Args:
        target_rnn: the target model, whose first layer defines the geometry.

    Returns:
        The per-head memory shape, excluding the leading head axis.
    """
    layers = getattr(target_rnn, "layers", None)
    if not layers:
        return (16, 16)
    first = layers[0][0]
    # Ask the layer for its own full state geometry, which accounts for targets
    # that stack several memories per head (HOPE stacks five).
    shape = getattr(first, "stacked_state_shape", None)
    if callable(shape):
        return tuple(shape())[1:]
    return (
        int(getattr(first, "key_dim", 16)),
        int(getattr(first, "value_dim", 16)),
    )


def _target_state_rms(target_rnn: nn.Module) -> float:
    """RMS magnitude of the target's own initial recurrent state.

    The translator ends in an RMSNorm, so its boundary state always has RMS 1.
    A target whose state lives at a different magnitude (HOPE's memories
    initialize near 0.02) would then start every non-first chunk far outside
    the basin the recurrence was trained in. Feeding the target's own initial
    RMS through as a fixed scale keeps both ends on the same scale.

    Args:
        target_rnn: the target model.

    Returns:
        The initial-state RMS, or 1.0 when it cannot be determined.
    """
    layers = getattr(target_rnn, "layers", None)
    if not layers:
        return 1.0
    init = getattr(layers[0][0], "init_state", None)
    if not callable(init):
        return 1.0
    try:
        hidden = init(1, torch.device("cpu")).hidden
    except (TypeError, RuntimeError):
        return 1.0
    with torch.no_grad():
        if not torch.isfinite(hidden).all():
            return 1.0
        rms = float(hidden.pow(2).mean().sqrt())
    return rms if rms > 0 else 1.0


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01
):
    """Create cosine annealing schedule with linear warmup."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def _clip_params(params, bound: float) -> None:
    for param in params:
        if param.grad is not None:
            param.grad.data.clamp_(-bound, bound)


def configure_gradient_clipping(
    model, target_grad_clip=1.0, scaffold_grad_clip=0.5, translator_grad_clip=0.5
):
    """Clamp gradients element-wise to per-group bounds before each optimizer step.

    Despite the ``*_clip`` names this is value clamping (``grad.clamp_``),
    not norm clipping. Applied via ``GradientClippingCallback``.

    Args:
        model: The ParallelRNNTrainer model
        target_grad_clip: Element-wise clamp bound for target RNN gradients
        scaffold_grad_clip: Element-wise clamp bound for scaffold gradients
        translator_grad_clip: Element-wise clamp bound for translator gradients
    """

    def clip_gradients():
        _clip_params(model.target.parameters(), target_grad_clip)
        for scaffold in model.scaffolds:
            _clip_params(scaffold.parameters(), scaffold_grad_clip)
        for translator in model.translators:
            _clip_params(translator.parameters(), translator_grad_clip)

    return clip_gradients


# Sentinel to detect if state was explicitly passed
_STATE_NOT_PROVIDED: object = object()


def _reshape_for_chunks(x: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, int, int, int]:
    """Reshape [B, T, D] -> [M*B, C, D] for parallel chunk processing.

    Returns:
        x_chunks: [M*B, C, D]
        num_chunks: M
        chunk_size: C (the actual chunk size used, which equals chunk_size after padding)
        padded_len: the padded sequence length (num_chunks * chunk_size)
    """
    batch, seq_len, feat_dim = x.shape
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    pad_len = num_chunks * chunk_size - seq_len
    if pad_len > 0:
        x = torch.cat(
            [x, torch.zeros(batch, pad_len, feat_dim, device=x.device, dtype=x.dtype)], dim=1
        )
    x_chunks = (
        x.view(batch, num_chunks, chunk_size, feat_dim)
        .transpose(0, 1)
        .reshape(num_chunks * batch, chunk_size, feat_dim)
    )
    padded_len = num_chunks * chunk_size
    return x_chunks, num_chunks, chunk_size, padded_len


def _reshape_from_chunks(
    x_chunks: torch.Tensor, batch: int, seq_len: int, num_chunks: int
) -> torch.Tensor:
    """Reshape [M*B, C, D] -> [B, T, D]."""
    x = x_chunks.view(num_chunks, batch, x_chunks.shape[1], x_chunks.shape[2]).transpose(0, 1)
    return x.reshape(batch, seq_len, x_chunks.shape[2])


def _unchunk_extra(
    extra: dict[str, torch.Tensor] | None, num_chunks: int, batch_size: int
) -> dict[str, torch.Tensor] | None:
    """Keep only the last chunk's entries from chunked extra state."""
    if extra is None or not isinstance(extra, dict):
        return None
    new_extra: dict[str, torch.Tensor] = {}
    for k, v in extra.items():
        if v.dim() >= 2 and v.shape[0] == num_chunks * batch_size:
            new_shape = (num_chunks, batch_size) + v.shape[1:]
            new_extra[k] = v.view(new_shape)[-1]
        else:
            new_extra[k] = v
    return new_extra


def _infer_target_type(target_rnn: nn.Module) -> str:
    """Infer the wrapper target_type (cell kind) from the target model.

    Handles NanoRNN (``mixer_type``) and the legacy layer classes
    (MinGRULayer / MinLSTMLayer / M2RNNLayer / MultiHeadRNNLayer).
    """
    mixer_type = getattr(target_rnn, "mixer_type", None)
    if mixer_type is not None:
        return str(mixer_type)
    layers = getattr(target_rnn, "layers", None)
    if layers is not None and len(layers) > 0:
        first = layers[0][0] if hasattr(layers[0], "__getitem__") else layers[0]
        name = type(first).__name__
        if "MinGRU" in name:
            return "min_gru"
        if "MinLSTM" in name:
            return "min_lstm"
        if "M2RNN" in name:
            return "m2rnn"
        heads = getattr(first, "heads", None)
        if heads is not None and len(heads) > 0 and "RKAN" in type(heads[0]).__name__:
            return "rkan"
    return "mlp"


class ParallelRNNTrainer(nn.Module):
    """Wrapper that enables parallel training of any RNN via chunkwise decomposition.

    During training (per layer):
    - Scaffold scans the layer's input to produce boundary summaries
    - Translator maps summaries to target boundary states
    - Target RNN layer runs in parallel across chunks from boundary states

    During inference:
    - Runs target RNN sequentially (scaffold and translator discarded)

    The wrapper exposes the final hidden states so custom output heads and losses
    can be attached for different sequence modeling objectives.
    """

    def __init__(
        self,
        target_rnn: nn.Module,
        chunk_size: int,
        scaffold_dim: int,
        target_type: str | None = None,
        num_layers: int | None = None,
        hidden_dim: int | None = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        detach_boundary: bool = False,
        use_gdn2_init: bool = True,
        output_head: Callable[[torch.Tensor], torch.Tensor] | None = None,
        scaffold_type: str = "min_gru",
        scaffold_num_layers: int = 1,
        scaffold_positional: bool = True,
        translator_type: str | None = None,
        translator_num_layers: int = 1,
        translator_pre_norm: bool = True,
        translator_norm_type: str = "rmsnorm",
        target_chunk_size: int | None = None,
    ) -> None:
        """Args:
        target_rnn: target RNN with ``layers`` as [rnn, ff] pairs.
        target_type: target cell kind (min_gru | min_lstm | mlp | rkan |
            faster_kan | m2rnn), inferred from the target when None.
            Drives state geometry (vector vs matrix), log-space positivity
            handling, and the default translator.
        detach_boundary: detach translated boundary states, cutting
            scaffold/translator gradients from the target path. Keep False:
            no auxiliary boundary loss exists, so True freezes the
            scaffold/translator at init.
        scaffold_type: min_gru | min_lstm | linear_sum | fourier.
        scaffold_num_layers: stacked scaffold layers per outer layer (1..N).
        scaffold_positional: add sinusoidal positional encoding to scaffold
            input (Fourier-Wavelet "X + P"); off is the FNet no-PE ablation.
        translator_type: mlp | rkan, or None to resolve from target_type.
        translator_num_layers: stacked translator layers per outer layer (1..N).
        translator_pre_norm: pre-normalize translator input (default True).
        translator_norm_type: rmsnorm | layernorm (default rmsnorm).
        target_chunk_size: Eq. 90's inner chunk length for self-referential
            matrix targets (HOPE), independent of ``chunk_size``. Within a
            chunk, Eq. 95's self-generated targets are frozen at the chunk
            start rather than read off the running memory, which is what stops
            that feedback loop from running away. Signals and retrieval stay on
            the running memory, so the cell trains the same function
            :meth:`step` runs. Must be strictly below ``chunk_size``; equal to
            it there is a single evaluation point and Eq. 90 degenerates to the
            sequential form. None (default) uses ``chunk_size``. Ignored by
            targets whose target is not self-generated (M2RNN).
        """
        super().__init__()
        if target_type is None:
            target_type = _infer_target_type(target_rnn)
        self.target = target_rnn
        self.chunk_size = chunk_size
        self.scaffold_dim = scaffold_dim
        self.detach_boundary = detach_boundary
        self.use_gdn2_init = use_gdn2_init
        self.target_type = target_type
        self.scaffold_type = scaffold_type
        self.scaffold_num_layers = scaffold_num_layers
        self.translator_num_layers = translator_num_layers
        self.hope_chunk_size = target_chunk_size

        # Optional output head: takes final hidden states [B, T, D] -> logits [B, T, C]
        # If None, use target's final_norm + classifier if available
        if output_head is not None:
            self.output_head = output_head
        else:
            final_norm = getattr(target_rnn, "final_norm", nn.Identity())
            classifier = getattr(target_rnn, "classifier", nn.Identity())
            self.output_head = nn.Sequential(final_norm, classifier)

        if num_layers is None:
            num_layers = getattr(target_rnn, "num_layers", 1)
        if hidden_dim is None:
            hidden_dim = getattr(target_rnn, "hidden_dim", scaffold_dim)
        if num_heads is None or num_heads == 4:  # Default value, infer from target
            num_heads = getattr(target_rnn, "num_heads", 4)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Per-layer scaffold stacks (each scans its layer's input)
        self.scaffolds = nn.ModuleList(
            [
                ScaffoldStack(
                    scaffold_type=scaffold_type,
                    input_dim=hidden_dim,  # Each layer's input is hidden_dim
                    hidden_dim=scaffold_dim,
                    num_layers=scaffold_num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                    positional=scaffold_positional,
                )
                for _ in range(num_layers)
            ]
        )

        # Per-layer translator stacks. translator_type is a pure hyperparameter;
        # output geometry (vector vs matrix) still derives from the target.
        resolved_translator = translator_type or _LEGACY_TARGET_TO_TRANSLATOR.get(
            target_type, "mlp"
        )
        self.translator_type = resolved_translator
        output_shape = "matrix" if target_type in MATRIX_TARGET_TYPES else "vector"
        # Matrix targets need the translator to emit one flat vector per head
        # that reshapes into the layer's memory geometry.
        translator_kwargs = {
            "input_dim": scaffold_dim,
            "hidden_dim": hidden_dim,
            "translator_type": resolved_translator,
            "num_layers": translator_num_layers,
            "output_shape": output_shape,
            "dropout": dropout,
            "use_gdn2_init": use_gdn2_init,
            "pre_norm": translator_pre_norm,
            "norm_type": translator_norm_type,
        }
        if output_shape == "matrix":
            geometry = _layer_memory_shape(target_rnn)
            # The layer state is [B, H, S, K, V]. The translator reshapes to
            # [B, M, H, S, K, V], so pass the geometry in that axis order.
            translator_kwargs.update(
                {
                    "num_heads": getattr(target_rnn, "num_heads", num_heads),
                    "num_memories": int(geometry[0]) if len(geometry) == 3 else 1,
                    "key_dim": int(geometry[-2]),
                    "value_dim": int(geometry[-1]),
                    "state_scale": _target_state_rms(target_rnn),
                }
            )
        self.translators = nn.ModuleList(
            [TranslatorStack(**translator_kwargs) for _ in range(num_layers)]
        )

        # Input projection to hidden_dim (if needed) for first layer.
        # Targets that own their embedding (e.g. NanoRNN.wte) already emit
        # hidden_dim, so the dummy input_dim must not create a bogus proj.
        self.input_proj = nn.Identity()
        if hasattr(target_rnn, "input_dim"):
            owns_embedding = hasattr(target_rnn, "wte") or hasattr(target_rnn, "_embed")
            input_dim = getattr(target_rnn, "input_dim", hidden_dim)
            if not owns_embedding and isinstance(input_dim, int) and input_dim != hidden_dim:
                self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: RNNStateList | None | object = _STATE_NOT_PROVIDED,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, RNNStateList] | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: parallel chunkwise in training, sequential in inference.

        Args:
            x: Input [B, T, D]
            state: Optional initial states
            return_hidden: If True, also return final hidden states [B, T, D]

        Returns:
            If state was explicitly passed (including None): (logits, new_state) -
            compatible with RNNModule protocol
            If state not provided and not training and return_hidden=False: logits [B, T, C]
            If state not provided and not training and return_hidden=True: (logits, hidden_states)
            If state not provided but training: (logits, new_state) for RNNModule compatibility
        """
        state_provided = state is not _STATE_NOT_PROVIDED
        state_list: RNNStateList | None = (
            cast(RNNStateList | None, state) if state_provided else None
        )

        # Accept token IDs from models that own their embedding (e.g. NanoRNN).
        if x.dim() == 2 and x.dtype == torch.long:
            embed = getattr(self.target, "_embed", None)
            if embed is None:
                raise ValueError(
                    "ParallelRNNTrainer received integer token IDs but target has no embedding"
                )
            x = embed(x)

        if not self.training:
            if return_hidden:
                return self._eval_with_hidden(x, state_list)
            out, new_state = self.target(x, state_list)
            if state_provided:
                return out, new_state
            return out

        batch, seq_len, _ = x.shape
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        # Scanner state at index t has consumed input t, so chunk m starts
        # from the state before its first token (index m*chunk_size - 1).
        boundary_indices = [m * self.chunk_size - 1 for m in range(1, num_chunks)]
        if not boundary_indices:
            warnings.warn(
                f"chunk_size ({self.chunk_size}) >= seq_len ({seq_len}): single-chunk "
                "fallback, scaffold/translator skipped; set chunk_size < T",
                UserWarning,
                stacklevel=2,
            )

        # Project input to hidden_dim only when it actually needs it.
        if (
            not isinstance(self.input_proj, nn.Identity)
            and x.shape[-1] == self.input_proj.in_features
            and x.shape[-1] != self.hidden_dim
        ):
            x = self.input_proj(x)

        layer_input = x
        current_state = state_list

        for layer_idx in range(self.num_layers):
            rnn_layer, ff_layer = self._get_layer_pair(layer_idx)

            if not boundary_indices:
                # Single chunk - run layer sequentially (scaffold/translator skipped)
                layer_input, current_state = self._run_layer_sequential(
                    layer_idx, layer_input, current_state
                )
                continue

            scaffold_states = self.scaffolds[layer_idx](layer_input)  # [B, T, scaffold_dim]

            boundary_scaffold = scaffold_states[:, boundary_indices, :]  # [B, M-1, scaffold_dim]

            boundary_states = self.translators[layer_idx](boundary_scaffold)  # [B, M-1, hidden_dim]

            # Optional: detach boundary states to isolate gradient paths
            if self.detach_boundary:
                boundary_states = boundary_states.detach()

            layer_output, layer_final_state = self._run_rnn_layer_parallel(
                layer_idx, rnn_layer, layer_input, boundary_states, current_state
            )

            layer_output = ff_layer(layer_output)

            if current_state is None:
                # Bootstrap the whole stack with each layer's own initial state.
                # Zeros here are wrong for a parameter-seeded layer: layer L+1
                # would start from a zero memory and never receive gradient.
                current_state = RNNStateList(
                    [
                        self._initial_state_for(
                            i, layer_input.shape[0], layer_input.device, layer_input.dtype
                        )
                        for i in range(self.num_layers)
                    ]
                )
            new_states = list(current_state.states)
            new_states[layer_idx] = layer_final_state
            current_state = RNNStateList(new_states)

            layer_input = layer_output

        hidden_states = layer_input
        assert current_state is not None

        if self.output_head is not None:
            logits = self.output_head(hidden_states)
            if state_provided:
                return logits, current_state
            if return_hidden:
                return logits, hidden_states
            return logits

        if return_hidden:
            return hidden_states, hidden_states
        return hidden_states

    def _eval_with_hidden(
        self, x: torch.Tensor, state: RNNStateList | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the target sequentially, capturing pre-head hidden states.

        Hooks the target's ``final_norm`` input so eval ``return_hidden``
        matches the training path (pre-head activations): callers can
        recompute ``logits == head(hidden)`` in either mode.
        """
        head = getattr(self.target, "final_norm", None)
        if head is None:
            head = getattr(self.target, "classifier", None)
        if head is None:
            raise AttributeError("return_hidden=True needs a target with final_norm or classifier")
        captured: dict[str, torch.Tensor] = {}

        def _capture(
            _module: nn.Module, args: tuple[torch.Tensor, ...], _output: torch.Tensor
        ) -> None:
            captured["hidden"] = args[0]

        handle = head.register_forward_hook(_capture)
        try:
            out, _ = self.target(x, state)
        finally:
            handle.remove()
        if "hidden" not in captured:
            raise RuntimeError(
                "target forward did not invoke final_norm/classifier; cannot capture hidden states"
            )
        return out, captured["hidden"]

    def training_step(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        state: RNNStateList | None = None,
        **loss_kwargs,
    ) -> torch.Tensor:
        """Execute one training step with a custom loss function.

        Args:
            x: Input [B, T, D]
            targets: Target labels/values [B, T, ...] or [B, ...]
            loss_fn: Callable(logits, targets, **kwargs) -> scalar loss
            state: Optional initial states
            **loss_kwargs: Additional arguments passed to loss_fn

        Returns:
            Scalar loss tensor
        """
        output = self.forward(x, state, return_hidden=False)
        logits = output[0] if isinstance(output, tuple) else output
        loss = loss_fn(logits, targets, **loss_kwargs)
        return loss

    @staticmethod
    def _layer_seeds_from_params(rnn_layer: nn.Module) -> bool:
        """Whether a layer's zero initial state would break its gradient chain.

        HOPE seeds its five memories from learned parameters, so a zero start
        makes ``M_v`` retrieve ``v_t = 0`` and the whole self-referential chain
        receives no gradient. M2RNN's recurrence is well defined at zero, so it
        keeps the cheaper zero start.
        """
        init = getattr(rnn_layer, "init_state", None)
        if not callable(init):
            return False
        # A layer that declares a parameter-seeded memory set exposes it; the
        # test is whether its init state is the trained parameter tensor.
        return any(name.endswith("_memory") for name, _ in rnn_layer.named_parameters())

    def _get_layer_pair(self, layer_idx: int) -> tuple[nn.Module, nn.Module]:
        """Get RNN layer and FF layer for given index."""
        layers = self.target.layers
        layer_pair = layers[layer_idx]  # type: ignore[index]
        if isinstance(layer_pair, nn.ModuleList):
            return layer_pair[0], layer_pair[1]
        return layer_pair[0], layer_pair[1]  # type: ignore[index, return-value]

    def _run_layer_sequential(
        self,
        layer_idx: int,
        layer_input: torch.Tensor,
        state: RNNStateList | None,
    ) -> tuple[torch.Tensor, RNNStateList | None]:
        """Run a single layer sequentially (for single chunk or inference)."""
        rnn_layer, ff_layer = self._get_layer_pair(layer_idx)

        layer_state = None
        if state is not None:
            layer_state = state[layer_idx]
        else:
            batch = layer_input.shape[0]
            num_heads = getattr(rnn_layer, "num_heads", self.num_heads)
            head_dim = getattr(rnn_layer, "head_dim", self.hidden_dim // self.num_heads)

            if self.target_type == "hope":
                # HOPE seeds its memories from learned parameters; a zero state
                # would kill the self-referential gradient chain.
                init = getattr(rnn_layer, "init_state", None)
                if not callable(init):
                    raise TypeError("HOPE layer must implement init_state")
                layer_state = init(batch, layer_input.device)
            elif self.target_type == "m2rnn":
                key_dim = getattr(rnn_layer, "key_dim", 16)
                value_dim = getattr(rnn_layer, "value_dim", 16)
                proj_dim = num_heads * (key_dim + key_dim + value_dim)
                layer_state = RNNState(
                    hidden=torch.zeros(
                        batch,
                        num_heads,
                        key_dim,
                        value_dim,
                        device=layer_input.device,
                        dtype=layer_input.dtype,
                    ),
                    extra={
                        "conv_cache": torch.zeros(
                            batch, proj_dim, 3, device=layer_input.device, dtype=layer_input.dtype
                        )
                    },
                )
            elif self.target_type in ("min_gru", "min_lstm"):
                # Log-space recurrences require strictly positive initial states.
                layer_state = RNNState(
                    hidden=torch.ones(
                        batch,
                        num_heads,
                        head_dim,
                        device=layer_input.device,
                        dtype=layer_input.dtype,
                    )
                )
            else:
                layer_state = RNNState(
                    hidden=torch.zeros(
                        batch,
                        num_heads,
                        head_dim,
                        device=layer_input.device,
                        dtype=layer_input.dtype,
                    )
                )

        layer_output, new_state = rnn_layer(layer_input, layer_state)
        layer_output = ff_layer(layer_output)

        if state is not None:
            new_states = list(state.states)
            new_states[layer_idx] = new_state
            return layer_output, RNNStateList(new_states)
        else:
            # Layers not yet reached need a placeholder initial state of their
            # own, not zeros. A parameter-seeded layer (HOPE) must get its
            # learned init here too, for the same reason its first chunk does:
            # a zero state would silently drop its gradient chain.
            batch, device, dtype = layer_input.shape[0], layer_input.device, layer_input.dtype
            new_states = [
                self._initial_state_for(layer_idx, batch, device, dtype)
                for layer_idx in range(self.num_layers)
            ]
            new_states[layer_idx] = new_state
            return layer_output, RNNStateList(new_states)

    def _initial_state_for(
        self, layer_idx: int, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> RNNState:
        """Initial state a layer starts from when it has not run yet."""
        rnn_layer, _ = self._get_layer_pair(layer_idx)
        if self.target_type == "hope" and self._layer_seeds_from_params(rnn_layer):
            init = getattr(rnn_layer, "init_state", None)
            if not callable(init):
                raise TypeError(f"{type(rnn_layer).__name__} must implement init_state")
            state = init(batch_size, device)
            return RNNState(hidden=state.hidden.to(dtype))
        num_heads = getattr(rnn_layer, "num_heads", self.num_heads)
        head_dim = getattr(rnn_layer, "head_dim", self.hidden_dim // self.num_heads)
        key_dim = getattr(rnn_layer, "key_dim", 16)
        value_dim = getattr(rnn_layer, "value_dim", 16)
        if self.target_type == "m2rnn":
            hidden = torch.zeros(
                batch_size, num_heads, key_dim, value_dim, device=device, dtype=dtype
            )
        else:
            hidden = torch.zeros(batch_size, num_heads, head_dim, device=device, dtype=dtype)
        return RNNState(hidden=hidden)

    def _run_rnn_layer_parallel(
        self,
        layer_idx: int,
        rnn_layer: nn.Module,
        layer_input: torch.Tensor,
        boundary_states: torch.Tensor,  # [B, M-1, D] or [B, M-1, N, K, V]
        initial_state: RNNStateList | None,
    ) -> tuple[torch.Tensor, RNNState]:
        """Run RNN layer in parallel across chunks.

        Args:
            rnn_layer: The RNN sublayer (e.g., MultiHeadRNNLayer, MinGRULayer)
            layer_input: [B, T, D]
            boundary_states: [B, M-1, D] translated boundary states
            initial_state: Initial states for the layer

        Returns:
            Tuple of (layer output [B, T, D], final state for this layer)
        """
        # Handle matrix-state targets specially (matrix memories + conv cache)
        if self.target_type == "hope":
            return self._run_hope_layer_parallel(
                layer_idx, rnn_layer, layer_input, boundary_states, initial_state
            )
        if self.target_type in MATRIX_TARGET_TYPES:
            return self._run_m2rnn_layer_parallel(
                layer_idx, rnn_layer, layer_input, boundary_states, initial_state
            )

        batch, seq_len, feat_dim = layer_input.shape
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        chunked_input, num_chunks, _chunk_len, padded_len = _reshape_for_chunks(
            layer_input, self.chunk_size
        )

        # boundary_states is [B, M-1, D], prepend zeros for first chunk
        zero_boundary = torch.zeros(
            batch, 1, self.hidden_dim, device=boundary_states.device, dtype=boundary_states.dtype
        )
        all_boundaries = torch.cat([zero_boundary, boundary_states], dim=1)  # [B, M, D]

        # For min_gru/min_lstm, boundary states must be positive for log-space operations
        # Apply softplus to ensure positive states (zeros become small positive)
        if self.target_type in ("min_gru", "min_lstm"):
            all_boundaries = functional.softplus(all_boundaries)

        chunk_initial_states = all_boundaries.transpose(0, 1).reshape(
            num_chunks * batch, -1
        )  # [M*B, D]

        if self.target_type in ("mlp", "rkan") and self.num_heads > 1:
            # D = H * D_head
            d_head = self.hidden_dim // self.num_heads
            chunk_initial_states = chunk_initial_states.view(
                num_chunks * batch, self.num_heads, d_head
            )

        if initial_state is not None:
            # Use provided initial state for first chunk only
            h0 = initial_state[layer_idx].hidden
            if h0.dim() == 3:
                # Multi-head: [B, H, D_head] -> expand to [M*B, H, D_head]
                h0 = (
                    h0.unsqueeze(0)
                    .expand(num_chunks, -1, -1, -1)
                    .reshape(num_chunks * batch, *h0.shape[1:])
                )
                if chunk_initial_states.dim() == 2:
                    # Vector-state target (e.g. min_gru/min_lstm): flatten heads.
                    h0 = h0.reshape(num_chunks * batch, -1)
            else:
                h0 = h0.unsqueeze(0).expand(num_chunks, -1, -1).reshape(num_chunks * batch, -1)
            # Override first chunk's state with provided initial state
            chunk_initial_states[:batch] = h0[:batch]

        chunk_state = RNNState(hidden=chunk_initial_states)
        chunk_output, chunk_final_state = rnn_layer.forward(chunked_input, chunk_state)  # type: ignore[misc]

        layer_output = _reshape_from_chunks(chunk_output, batch, padded_len, num_chunks)

        if chunk_final_state.hidden.dim() == 2:
            # [M*B, D] -> reshape to [M, B, D] -> take last [B, D]
            final_hidden = chunk_final_state.hidden.view(num_chunks, batch, -1)[-1]  # [B, D]
        else:
            # [M*B, H, D_head] -> reshape to [M, B, H, D_head] -> take last [B, H, D_head]
            final_hidden = chunk_final_state.hidden.view(
                num_chunks, batch, *chunk_final_state.hidden.shape[1:]
            )[-1]

        final_state = RNNState(
            hidden=final_hidden,
            extra=_unchunk_extra(chunk_final_state.extra, num_chunks, batch),
        )

        return layer_output[:, :seq_len, :], final_state

    def _run_hope_layer_parallel(
        self,
        layer_idx: int,
        rnn_layer: nn.Module,
        layer_input: torch.Tensor,
        boundary_states: torch.Tensor,  # [B, M-1, H, 5, N, N]
        initial_state: RNNStateList | None,
    ) -> tuple[torch.Tensor, RNNState]:
        """Run a HOPE layer in parallel across chunks.

        HOPE gets its own path rather than sharing the M2RNN one because its
        state is not a single matrix: the layer carries five memories per head
        plus two pieces of *token* history that M2RNN has no equivalent of --
        the depthwise ``q`` conv cache and the Continuum Memory buffer.  Both are
        carried across chunks here, so a chunk resumes with the same recurrent
        context the sequential form would have had rather than restarting cold
        at every boundary.

        Inside a chunk the layer runs its own recurrence.  Eq. 95's
        self-generated targets are frozen at the chunk start
        (``target_chunk_size``) while every signal is evaluated at the running
        memory, which is what :meth:`HOPELayer.step` does at inference, so the
        training and decoding paths are the same function.

        Args:
            layer_idx: index of the outer layer.
            rnn_layer: the HOPELayer module.
            layer_input: [B, T, D]
            boundary_states: [B, M-1, H, 5, N, N] translated boundary memories.
            initial_state: optional initial states.

        Returns:
            Tuple of (layer output [B, T, D], final state for this layer).
        """
        batch, seq_len, _ = layer_input.shape
        num_heads = getattr(rnn_layer, "num_heads", self.num_heads)
        shape = getattr(rnn_layer, "stacked_state_shape", None)
        if not callable(shape):
            raise TypeError(f"{type(rnn_layer).__name__} must implement stacked_state_shape")
        mem_shape = tuple(shape())
        if mem_shape[0] != num_heads:
            raise ValueError(
                f"{type(rnn_layer).__name__}.stacked_state_shape() reports {mem_shape[0]} heads "
                f"but the wrapper was told {num_heads}"
            )

        chunked_input, num_chunks, _chunk_len, padded_len = _reshape_for_chunks(
            layer_input, self.chunk_size
        )

        # Chunk 0 seeds from the layer's own learned memories, never from the
        # translator: a zero M_v retrieves v_t = 0 and would zero the gradient of
        # the whole self-referential chain.
        seeded = getattr(rnn_layer, "init_state", None)
        if not callable(seeded):
            raise TypeError(f"{type(rnn_layer).__name__} must implement init_state")
        first_chunk_state = seeded(batch, layer_input.device).hidden
        if first_chunk_state.shape[1:] != torch.Size(mem_shape):
            raise ValueError(
                f"{type(rnn_layer).__name__}.init_state gave "
                f"{tuple(first_chunk_state.shape[1:])} but the layer reports "
                f"a memory shape of {tuple(mem_shape)}"
            )
        all_boundaries = torch.cat([first_chunk_state.unsqueeze(1), boundary_states], dim=1)

        # [B, M, *mem_shape] -> [M*B, *mem_shape]
        chunk_initial_states = all_boundaries.transpose(0, 1).reshape(
            num_chunks * batch, *mem_shape
        )

        # The conv cache and CMS buffer are token history. Per chunk they start
        # from the caller's own state (zeros for a fresh sequence) rather than
        # from the previous chunk, since a parallel scan cannot know what the
        # scaffold's boundary summary said about the tokens in between. Restarting
        # them is the same approximation M2RNN already makes, and it keeps each
        # chunk's state independent, which is what makes the batch dimension
        # below a genuine batch.
        cache_key = "q"
        extra: dict[str, torch.Tensor] = {}
        provided = initial_state[layer_idx].extra if initial_state is not None else None
        for key in (cache_key, "cms"):
            value = provided.get(key) if provided else None
            if value is not None:
                extra[key] = value.unsqueeze(0).expand(num_chunks, -1, *([-1] * (value.dim() - 1)))
                extra[key] = extra[key].reshape(num_chunks * batch, *value.shape[1:])

        if initial_state is not None:
            h0 = initial_state[layer_idx].hidden
            if h0.shape[1:] != torch.Size(mem_shape):
                raise ValueError(
                    f"initial state {tuple(h0.shape)} does not match the layer's "
                    f"memory shape {tuple(mem_shape)}"
                )
            h0_expanded = (
                h0.unsqueeze(0)
                .expand(num_chunks, -1, *([-1] * (h0.dim() - 1)))
                .reshape(num_chunks * batch, *mem_shape)
            )
            chunk_initial_states[:batch] = h0_expanded[:batch]

        chunk_state = RNNState(hidden=chunk_initial_states, extra=extra or None)

        target_kwargs = (
            {"chunk_size": self.hope_chunk_size}
            if self.hope_chunk_size and self.hope_chunk_size < self.chunk_size
            else {}
        )
        chunk_output, chunk_final_state = rnn_layer.forward(  # type: ignore[misc]
            chunked_input, chunk_state, **target_kwargs
        )

        layer_output = _reshape_from_chunks(chunk_output, batch, padded_len, num_chunks)
        final_hidden = chunk_final_state.hidden.view(num_chunks, batch, *mem_shape)[-1]
        final_extra = _unchunk_extra(chunk_final_state.extra, num_chunks, batch)

        return layer_output[:, :seq_len, :], RNNState(hidden=final_hidden, extra=final_extra)

    def _run_m2rnn_layer_parallel(
        self,
        layer_idx: int,
        rnn_layer: nn.Module,
        layer_input: torch.Tensor,
        boundary_states: torch.Tensor,  # [B, M-1, N, K, V]
        initial_state: RNNStateList | None,
    ) -> tuple[torch.Tensor, RNNState]:
        """Run an M2RNN layer in parallel across chunks.

        M2RNN carries one ``[N, K, V]`` matrix per head plus a causal conv
        cache that must be carried across chunks. HOPE, the other matrix-state
        target, has five memories per head and two pieces of token history, so
        it has its own :meth:`_run_hope_layer_parallel`.

        Args:
            layer_idx: index of the outer layer.
            rnn_layer: the M2RNNLayer module.
            layer_input: [B, T, D]
            boundary_states: [B, M-1, ...] translated boundary memories.
            initial_state: optional initial states.

        Returns:
            Tuple of (layer output [B, T, D], final state for this layer).
        """
        batch, seq_len, _ = layer_input.shape
        num_heads = getattr(rnn_layer, "num_heads", self.num_heads)
        # Ask the layer for its own state geometry rather than hard-coding
        # M2RNN's [H, K, V]. A layer that exposes none keeps the translator's own
        # geometry, which is the same tensor these boundaries came from.
        shape = getattr(rnn_layer, "stacked_state_shape", None)
        mem_shape = tuple(shape()) if callable(shape) else tuple(boundary_states.shape[2:])
        if mem_shape[0] != num_heads:
            raise ValueError(
                f"{type(rnn_layer).__name__}.state_shape() reports {mem_shape[0]} heads "
                f"but the wrapper was told {num_heads}"
            )

        chunked_input, num_chunks, _chunk_len, padded_len = _reshape_for_chunks(
            layer_input, self.chunk_size
        )

        # Chunk 0's init comes from the layer itself, not the translator, when a zero
        # start would break its gradient chain. M2RNN's recurrence is well
        # defined at zero, so it keeps the cheaper zero start, matching the
        # sequential path.
        if self._layer_seeds_from_params(rnn_layer):
            init = getattr(rnn_layer, "init_state", None)
            if not callable(init):
                raise TypeError(f"{type(rnn_layer).__name__} must implement init_state")
            # init_state returns the batched [B, *mem_shape] state; boundaries
            # carry a chunk axis, so insert it to concatenate.
            first_chunk_state = init(batch, layer_input.device).hidden
            if first_chunk_state.shape[1:] != torch.Size(mem_shape):
                raise ValueError(
                    f"{type(rnn_layer).__name__}.init_state gave "
                    f"{tuple(first_chunk_state.shape[1:])} but the layer reports "
                    f"a memory shape of {tuple(mem_shape)}"
                )
        else:
            first_chunk_state = torch.zeros(
                batch, *mem_shape, device=boundary_states.device, dtype=boundary_states.dtype
            )
        all_boundaries = torch.cat([first_chunk_state.unsqueeze(1), boundary_states], dim=1)

        # [B, M, *mem_shape] -> [M, B, *mem_shape] -> [M*B, *mem_shape]
        chunk_initial_states = all_boundaries.transpose(0, 1).reshape(
            num_chunks * batch, *mem_shape
        )

        cache_key = "conv_cache"
        chunk_cache: torch.Tensor | None = None
        if initial_state is not None:
            extra = initial_state[layer_idx].extra
            if extra is not None:
                provided = extra.get(cache_key)
                if provided is not None:
                    chunk_cache = (
                        provided.unsqueeze(0)
                        .expand(num_chunks, -1, *([-1] * (provided.dim() - 1)))
                        .reshape(num_chunks * batch, *provided.shape[1:])
                    )

            h0 = initial_state[layer_idx].hidden
            if h0.shape[1:] != torch.Size(mem_shape):
                raise ValueError(
                    f"initial state {tuple(h0.shape)} does not match the layer's "
                    f"memory shape {tuple(mem_shape)}"
                )
            h0_expanded = (
                h0.unsqueeze(0)
                .expand(num_chunks, -1, *([-1] * (h0.dim() - 1)))
                .reshape(num_chunks * batch, *mem_shape)
            )
            # Only chunk 0's rows come from h0; later chunks keep the
            # translator's boundary states. The cache needs no override: it was
            # just expanded from the caller's own tensor, so its leading rows
            # already hold it.
            chunk_initial_states[:batch] = h0_expanded[:batch]

        extra = {cache_key: chunk_cache} if chunk_cache is not None else None
        chunk_state = RNNState(hidden=chunk_initial_states, extra=extra)

        # HOPE routes to _run_hope_layer_parallel, so only M2RNN reaches here.
        # Its inner chunk is the scaffold chunk, which degenerates to the
        # sequential form; Eq. 90's frozen targets are a HOPE-only concern.
        chunk_output, chunk_final_state = rnn_layer.forward(  # type: ignore[misc]
            chunked_input, chunk_state
        )

        layer_output = _reshape_from_chunks(chunk_output, batch, padded_len, num_chunks)

        final_hidden = chunk_final_state.hidden.view(num_chunks, batch, *mem_shape)[-1]

        final_extra = _unchunk_extra(chunk_final_state.extra, num_chunks, batch)

        final_state = RNNState(hidden=final_hidden, extra=final_extra)

        return layer_output[:, :seq_len, :], final_state

    def get_boundary_states(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Get translated boundary states for all layers for inspection."""
        if (
            not isinstance(self.input_proj, nn.Identity)
            and x.shape[-1] == self.input_proj.in_features
            and x.shape[-1] != self.hidden_dim
        ):
            x = self.input_proj(x)
        batch, seq_len, _ = x.shape
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        # Aligned with forward: chunk m starts from the scanner state before
        # its first token.
        boundary_indices = [m * self.chunk_size - 1 for m in range(1, num_chunks)]

        if not boundary_indices:
            return [
                torch.empty(batch, 0, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]

        boundary_states_per_layer = []
        layer_input = x

        for layer_idx in range(self.num_layers):
            scaffold_states = self.scaffolds[layer_idx](layer_input)
            boundary_scaffold = scaffold_states[:, boundary_indices, :]
            boundary_states = self.translators[layer_idx](boundary_scaffold)
            boundary_states_per_layer.append(boundary_states)

        return boundary_states_per_layer

    def step(self, x_t: torch.Tensor, state: RNNStateList | None = None):
        """Inference step: runs target RNN sequentially (no scaffold/translator)."""
        step_fn = getattr(self.target, "step", None)
        if callable(step_fn):
            return step_fn(x_t, state)
        out, new_state = self.target(x_t.unsqueeze(1), state)
        return out.squeeze(1), new_state

    def train(self, mode: bool = True) -> ParallelRNNTrainer:
        """Set training mode; super().train() reaches every child module."""
        super().train(mode)
        return self

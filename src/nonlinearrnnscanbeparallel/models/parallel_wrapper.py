"""Parallel RNN Training Wrapper.

Implements the chunkwise parallel training framework:
1. Per-layer scaffold scans the layer's input to produce boundary summaries
2. Translator maps scaffold summaries to target RNN boundary states
3. Target RNN runs in parallel across chunks from boundary states
4. Gradients are isolated: target from within-chunk BPTT, scaffold/translator from boundary path

The wrapper exposes hidden states so custom output heads and losses can be attached.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from .base import RNNState, RNNStateList
from .scaffold import ScaffoldStack
from .translator import _LEGACY_TARGET_TO_TRANSLATOR, TranslatorStack


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
    """Configure gradient clipping for different parameter groups.

    Args:
        model: The ParallelRNNTrainer model
        target_grad_clip: Max gradient norm for target RNN parameters
        scaffold_grad_clip: Max gradient norm for scaffold parameters
        translator_grad_clip: Max gradient norm for translator parameters
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
        padded_T: the padded sequence length (num_chunks * chunk_size)
    """
    B, T, D = x.shape
    num_chunks = (T + chunk_size - 1) // chunk_size
    # Pad if needed
    pad_len = num_chunks * chunk_size - T
    if pad_len > 0:
        x = torch.cat([x, torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)], dim=1)
    # Reshape: [B, M*C, D] -> [B, M, C, D] -> [M*B, C, D]
    x_chunks = (
        x.view(B, num_chunks, chunk_size, D).transpose(0, 1).reshape(num_chunks * B, chunk_size, D)
    )
    padded_T = num_chunks * chunk_size
    return x_chunks, num_chunks, chunk_size, padded_T


def _reshape_from_chunks(x_chunks: torch.Tensor, B: int, T: int, num_chunks: int) -> torch.Tensor:
    """Reshape [M*B, C, D] -> [B, T, D]."""
    # [M*B, C, D] -> [M, B, C, D] -> [B, M, C, D] -> [B, T, D]
    x = x_chunks.view(num_chunks, B, x_chunks.shape[1], x_chunks.shape[2]).transpose(0, 1)
    return x.reshape(B, T, x_chunks.shape[2])


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
        translator_type: str | None = None,
        translator_num_layers: int = 1,
    ) -> None:
        """Args:
        target_rnn: target RNN with ``layers`` as [rnn, ff] pairs.
        target_type: deprecated legacy selector for the m2rnn code path
            and default translator. Prefer explicit scaffold/translator keys.
        scaffold_type: min_gru | min_lstm.
        scaffold_num_layers: stacked scaffold layers per outer layer (1..N).
        translator_type: mlp | rkan, or None to resolve from target_type.
        translator_num_layers: stacked translator layers per outer layer (1..N).
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

        # Optional output head: takes final hidden states [B, T, D] -> logits [B, T, C]
        # If None, use target's final_norm + classifier if available
        if output_head is not None:
            self.output_head = output_head
        else:
            final_norm = getattr(target_rnn, "final_norm", nn.Identity())
            classifier = getattr(target_rnn, "classifier", nn.Identity())
            self.output_head = nn.Sequential(final_norm, classifier)

        # Infer target properties if not provided
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
        output_shape = "matrix" if target_type == "m2rnn" else "vector"
        # For M2RNN, pass additional params (num_heads, key_dim, value_dim)
        translator_kwargs = {
            "input_dim": scaffold_dim,
            "hidden_dim": hidden_dim,
            "translator_type": resolved_translator,
            "num_layers": translator_num_layers,
            "output_shape": output_shape,
            "dropout": dropout,
            "use_gdn2_init": use_gdn2_init,
        }
        if target_type == "m2rnn":
            translator_kwargs.update(
                {
                    "num_heads": getattr(target_rnn, "num_heads", num_heads),
                    "key_dim": getattr(target_rnn, "key_dim", 16),
                    "value_dim": getattr(target_rnn, "value_dim", 16),
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
        # Check if state was explicitly passed
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

        # In inference mode, run target RNN sequentially (no scaffold/translator)
        if not self.training:
            out, new_state = self.target(x, state_list)
            if return_hidden:
                return out, out  # For seq models, output often IS the hidden state
            # Always return tuple for RNNModule compatibility when state was provided
            if state_provided:
                return out, new_state
            return out

        B, T, _ = x.shape
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        # Scanner state at index t has consumed input t, so chunk m starts
        # from the state before its first token (index m*chunk_size - 1).
        boundary_indices = [m * self.chunk_size - 1 for m in range(1, num_chunks)]
        if not boundary_indices:
            warnings.warn(
                f"chunk_size ({self.chunk_size}) >= seq_len ({T}): single-chunk "
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

        # Process layer by layer
        layer_input = x
        current_state = state_list

        for layer_idx in range(self.num_layers):
            # Get the RNN and FF sublayers for this layer
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

            # Update current_state with the final state from this layer
            if current_state is not None:
                # Update the state for this layer with the final state
                new_states = list(current_state.states)
                new_states[layer_idx] = layer_final_state
                current_state = RNNStateList(new_states)
            else:
                # Initialize current_state from layer_final_state for the first layer
                if layer_idx == 0:
                    new_states = [
                        RNNState(hidden=torch.zeros_like(layer_final_state.hidden))
                        for _ in range(self.num_layers)
                    ]
                    current_state = RNNStateList(new_states)
                assert current_state is not None
                new_states = list(current_state.states)
                new_states[layer_idx] = layer_final_state
                current_state = RNNStateList(new_states)

            # Prepare for next layer
            layer_input = layer_output

        # layer_input is now final hidden states [B, T, D]
        hidden_states = layer_input
        assert current_state is not None

        if self.output_head is not None:
            logits = self.output_head(hidden_states)
            # Return tuple if state was explicitly provided (for RNNModule compatibility)
            # or if in inference mode with state provided
            if state_provided:
                return logits, current_state
            if return_hidden:
                return logits, hidden_states
            return logits

        if return_hidden:
            return hidden_states, hidden_states
        return hidden_states

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

    def _get_layer_pair(self, layer_idx: int) -> tuple[nn.Module, nn.Module]:
        """Get RNN layer and FF layer for given index."""
        # Target RNNs in this codebase have self.layers as ModuleList of [rnn, ff] pairs
        layers = self.target.layers
        layer_pair = layers[layer_idx]  # type: ignore[index]
        if isinstance(layer_pair, nn.ModuleList):
            return layer_pair[0], layer_pair[1]
        # Fallback: assume it's a tuple/list
        return layer_pair[0], layer_pair[1]  # type: ignore[index, return-value]

    def _run_layer_sequential(
        self,
        layer_idx: int,
        layer_input: torch.Tensor,
        state: RNNStateList | None,
    ) -> tuple[torch.Tensor, RNNStateList | None]:
        """Run a single layer sequentially (for single chunk or inference)."""
        rnn_layer, ff_layer = self._get_layer_pair(layer_idx)

        # Initialize state if None - the individual RNN layer needs a state
        layer_state = None
        if state is not None:
            layer_state = state[layer_idx]
        else:
            # Create initial state for this layer
            B = layer_input.shape[0]
            # Get dimensions safely
            num_heads = getattr(rnn_layer, "num_heads", self.num_heads)
            head_dim = getattr(rnn_layer, "head_dim", self.hidden_dim // self.num_heads)

            if self.target_type == "m2rnn":
                key_dim = getattr(rnn_layer, "key_dim", 16)
                value_dim = getattr(rnn_layer, "value_dim", 16)
                proj_dim = num_heads * (key_dim + key_dim + value_dim)
                layer_state = RNNState(
                    hidden=torch.zeros(
                        B,
                        num_heads,
                        key_dim,
                        value_dim,
                        device=layer_input.device,
                        dtype=layer_input.dtype,
                    ),
                    extra={
                        "conv_cache": torch.zeros(
                            B, proj_dim, 3, device=layer_input.device, dtype=layer_input.dtype
                        )
                    },
                )
            elif self.target_type in ("min_gru", "min_lstm"):
                # Log-space recurrences require strictly positive initial states.
                layer_state = RNNState(
                    hidden=torch.ones(
                        B,
                        num_heads,
                        head_dim,
                        device=layer_input.device,
                        dtype=layer_input.dtype,
                    )
                )
            else:
                layer_state = RNNState(
                    hidden=torch.zeros(
                        B,
                        num_heads,
                        head_dim,
                        device=layer_input.device,
                        dtype=layer_input.dtype,
                    )
                )

        layer_output, new_state = rnn_layer(layer_input, layer_state)
        layer_output = ff_layer(layer_output)

        # Return full state list with this layer's new state
        if state is not None:
            new_states = list(state.states)
            new_states[layer_idx] = new_state
            return layer_output, RNNStateList(new_states)
        else:
            # Create new state list
            new_states = [
                RNNState(hidden=torch.zeros_like(new_state.hidden)) for _ in range(self.num_layers)
            ]
            new_states[layer_idx] = new_state
            return layer_output, RNNStateList(new_states)

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
        # Handle M2RNN specially due to matrix state and conv_cache
        if self.target_type == "m2rnn":
            return self._run_m2rnn_layer_parallel(
                layer_idx, rnn_layer, layer_input, boundary_states, initial_state
            )

        B, T, D = layer_input.shape
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size

        # Reshape input for parallel chunk processing: [B, T, D] -> [M*B, C, D]
        chunked_input, num_chunks, C, padded_T = _reshape_for_chunks(layer_input, self.chunk_size)

        # Expand boundary states to per-chunk initial states: [B, M, D] -> [M*B, D]
        # boundary_states is [B, M-1, D], prepend zeros for first chunk
        zero_boundary = torch.zeros(
            B, 1, self.hidden_dim, device=boundary_states.device, dtype=boundary_states.dtype
        )
        all_boundaries = torch.cat([zero_boundary, boundary_states], dim=1)  # [B, M, D]

        # For min_gru/min_lstm, boundary states must be positive for log-space operations
        # Apply softplus to ensure positive states (zeros become small positive)
        if self.target_type in ("min_gru", "min_lstm"):
            all_boundaries = F.softplus(all_boundaries)

        chunk_initial_states = all_boundaries.transpose(0, 1).reshape(
            num_chunks * B, -1
        )  # [M*B, D]

        # For multi-head RNNs, reshape to [M*B, H, D_head]
        if self.target_type in ("mlp", "rkan") and self.num_heads > 1:
            # D = H * D_head
            d_head = self.hidden_dim // self.num_heads
            chunk_initial_states = chunk_initial_states.view(num_chunks * B, self.num_heads, d_head)

        # Get initial state for this layer
        if initial_state is not None:
            # Use provided initial state for first chunk only
            h0 = initial_state[layer_idx].hidden
            if h0.dim() == 3:
                # Multi-head: [B, H, D_head] -> expand to [M*B, H, D_head]
                h0 = (
                    h0.unsqueeze(0)
                    .expand(num_chunks, -1, -1, -1)
                    .reshape(num_chunks * B, *h0.shape[1:])
                )
                if chunk_initial_states.dim() == 2:
                    # Vector-state target (e.g. min_gru/min_lstm): flatten heads.
                    h0 = h0.reshape(num_chunks * B, -1)
            else:
                # Single vector: [B, D] -> expand to [M*B, D]
                h0 = h0.unsqueeze(0).expand(num_chunks, -1, -1).reshape(num_chunks * B, -1)
            # Override first chunk's state with provided initial state
            chunk_initial_states[:B] = h0[:B]

        # Run RNN layer on chunked input
        # The RNN layer's forward expects [B, T, D] and state
        # We give it [M*B, C, D] with expanded states
        chunk_state = RNNState(hidden=chunk_initial_states)
        chunk_output, chunk_final_state = rnn_layer.forward(chunked_input, chunk_state)  # type: ignore[misc]

        # Reshape back: [M*B, C, D] -> [B, T, D]
        layer_output = _reshape_from_chunks(chunk_output, B, padded_T, num_chunks)

        # Extract final state from last chunk
        # chunk_final_state.hidden is [M*B, D] or [M*B, H, D_head]
        # We need the last chunk's state: [B, D] or [B, H, D_head]
        if chunk_final_state.hidden.dim() == 2:
            # [M*B, D] -> reshape to [M, B, D] -> take last [B, D]
            final_hidden = chunk_final_state.hidden.view(num_chunks, B, -1)[-1]  # [B, D]
        else:
            # [M*B, H, D_head] -> reshape to [M, B, H, D_head] -> take last [B, H, D_head]
            final_hidden = chunk_final_state.hidden.view(
                num_chunks, B, *chunk_final_state.hidden.shape[1:]
            )[-1]

        final_state = RNNState(hidden=final_hidden)
        if chunk_final_state.extra is not None:
            # Handle extra (e.g., conv_cache) - take from last chunk
            extra = chunk_final_state.extra
            if isinstance(extra, dict):
                new_extra = {}
                for k, v in extra.items():
                    if v.dim() >= 2 and v.shape[0] == num_chunks * B:
                        # Reshape and take last chunk
                        new_shape = (num_chunks, B) + v.shape[1:]
                        new_extra[k] = v.view(new_shape)[-1]
                    else:
                        new_extra[k] = v
                final_state = RNNState(hidden=final_hidden, extra=new_extra)

        # Trim padding if any
        return layer_output[:, :T, :], final_state

    def _run_m2rnn_layer_parallel(
        self,
        layer_idx: int,
        rnn_layer: nn.Module,
        layer_input: torch.Tensor,
        boundary_states: torch.Tensor,  # [B, M-1, N, K, V]
        initial_state: RNNStateList | None,
    ) -> tuple[torch.Tensor, RNNState]:
        """Run M2RNN layer in parallel across chunks.

        M2RNN has matrix state [B, N, K, V] and conv_cache that must be
        properly carried across chunks.

        Returns:
            Tuple of (layer output [B, T, D], final state for this layer)
        """
        B, T, D = layer_input.shape
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size

        # Get M2RNN dimensions from the layer
        num_heads = getattr(rnn_layer, "num_heads", self.num_heads)
        key_dim = getattr(rnn_layer, "key_dim", 16)
        value_dim = getattr(rnn_layer, "value_dim", 16)

        # Reshape input for parallel chunk processing: [B, T, D] -> [M*B, C, D]
        chunked_input, num_chunks, C, padded_T = _reshape_for_chunks(layer_input, self.chunk_size)

        # Expand boundary states to per-chunk initial states
        # boundary_states is [B, M-1, N, K, V], prepend zeros for first chunk
        zero_boundary = torch.zeros(
            B,
            1,
            num_heads,
            key_dim,
            value_dim,
            device=boundary_states.device,
            dtype=boundary_states.dtype,
        )
        all_boundaries = torch.cat([zero_boundary, boundary_states], dim=1)  # [B, M, N, K, V]

        # Expand to chunks: [B, M, N, K, V] -> [M, B, N, K, V] -> [M*B, N, K, V]
        chunk_initial_states = all_boundaries.transpose(0, 1).reshape(
            num_chunks * B, num_heads, key_dim, value_dim
        )  # [M*B, N, K, V]

        # Handle conv_cache if provided in initial_state
        chunk_conv_cache = None
        if initial_state is not None:
            extra = initial_state[layer_idx].extra
            if extra is not None:
                conv_cache = extra.get("conv_cache")
                if conv_cache is not None:
                    # conv_cache is [B, proj_dim, kernel_size-1]
                    # Expand to chunks: [B, ...] -> [M*B, ...]
                    chunk_conv_cache = (
                        conv_cache.unsqueeze(0)
                        .expand(num_chunks, -1, -1, -1)
                        .reshape(num_chunks * B, *conv_cache.shape[1:])
                    )

        # Override first chunk's state with provided initial state
        if initial_state is not None:
            h0 = initial_state[layer_idx].hidden
            # h0 is [B, N, K, V], expand to [M*B, N, K, V]
            h0 = (
                h0.unsqueeze(0)
                .expand(num_chunks, -1, -1, -1, -1)
                .reshape(num_chunks * B, num_heads, key_dim, value_dim)
            )
            chunk_initial_states[:B] = h0[:B]

            # Also use provided conv_cache for first chunk
            if chunk_conv_cache is not None:
                extra = initial_state[layer_idx].extra
                if extra is not None:
                    provided_cache = extra.get("conv_cache")
                    if provided_cache is not None:
                        chunk_conv_cache[:B] = provided_cache

        # Create state with hidden and conv_cache
        extra = {"conv_cache": chunk_conv_cache} if chunk_conv_cache is not None else None
        chunk_state = RNNState(hidden=chunk_initial_states, extra=extra)

        # Run RNN layer on chunked input
        chunk_output, chunk_final_state = rnn_layer.forward(chunked_input, chunk_state)  # type: ignore[misc]

        # Reshape back: [M*B, C, D] -> [B, T, D]
        layer_output = _reshape_from_chunks(chunk_output, B, padded_T, num_chunks)

        # Extract final state from last chunk
        # chunk_final_state.hidden is [M*B, N, K, V]
        final_hidden = chunk_final_state.hidden.view(num_chunks, B, num_heads, key_dim, value_dim)[
            -1
        ]  # [B, N, K, V]

        final_extra = None
        if chunk_final_state.extra is not None:
            extra = chunk_final_state.extra
            if isinstance(extra, dict):
                new_extra = {}
                for k, v in extra.items():
                    if v.dim() >= 2 and v.shape[0] == num_chunks * B:
                        # Reshape and take last chunk
                        new_shape = (num_chunks, B) + v.shape[1:]
                        new_extra[k] = v.view(new_shape)[-1]
                    else:
                        new_extra[k] = v
                final_extra = new_extra

        final_state = RNNState(hidden=final_hidden, extra=final_extra)

        # Trim padding if any
        return layer_output[:, :T, :], final_state

    def get_boundary_states(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Get translated boundary states for all layers for inspection."""
        if (
            not isinstance(self.input_proj, nn.Identity)
            and x.shape[-1] == self.input_proj.in_features
            and x.shape[-1] != self.hidden_dim
        ):
            x = self.input_proj(x)
        B, T, _ = x.shape
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        # Aligned with forward: chunk m starts from the scanner state before
        # its first token.
        boundary_indices = [m * self.chunk_size - 1 for m in range(1, num_chunks)]

        if not boundary_indices:
            return [
                torch.empty(B, 0, self.hidden_dim, device=x.device) for _ in range(self.num_layers)
            ]

        boundary_states_per_layer = []
        layer_input = x

        for layer_idx in range(self.num_layers):
            scaffold_states = self.scaffolds[layer_idx](layer_input)
            boundary_scaffold = scaffold_states[:, boundary_indices, :]
            boundary_states = self.translators[layer_idx](boundary_scaffold)
            boundary_states_per_layer.append(boundary_states)

            # For next layer, we'd need the layer output
            # This is just for inspection, so we skip the actual layer computation
            # In practice, you'd run the full forward to get accurate boundary states

        return boundary_states_per_layer

    def step(self, x_t: torch.Tensor, state: RNNStateList | None = None):
        """Inference step: runs target RNN sequentially (no scaffold/translator)."""
        step_fn = getattr(self.target, "step", None)
        if callable(step_fn):
            return step_fn(x_t, state)
        # Fallback: run single-step forward
        return self.target(x_t.unsqueeze(1), state)[0].squeeze(1), state

    def train(self, mode: bool = True) -> ParallelRNNTrainer:
        """Set training mode."""
        self.training = mode
        self.target.train(mode)
        for scaffold in self.scaffolds:
            scaffold.train(mode)
        for trans in self.translators:
            trans.train(mode)
        return self

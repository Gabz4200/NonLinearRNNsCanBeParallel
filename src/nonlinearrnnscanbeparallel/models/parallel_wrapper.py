"""Parallel RNN Training Wrapper.

Implements the chunkwise parallel training framework:
1. Per-layer scaffold scans the layer's input to produce boundary summaries
2. Translator maps scaffold summaries to target RNN boundary states
3. Target RNN runs in parallel across chunks from boundary states
4. Gradients are isolated: target from within-chunk BPTT, scaffold/translator from boundary path
"""

from __future__ import annotations

import torch
from torch import nn

from .base import RNNState, RNNStateList
from .scaffold import MinGRUScaffold
from .translator import Translator


def _reshape_for_chunks(x: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, int, int]:
    """Reshape [B, T, D] -> [M*B, C, D] for parallel chunk processing.

    Returns:
        x_chunks: [M*B, C, D]
        num_chunks: M
        chunk_size: C (may be smaller for last chunk)
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
    return x_chunks, num_chunks, chunk_size


def _reshape_from_chunks(x_chunks: torch.Tensor, B: int, T: int, num_chunks: int) -> torch.Tensor:
    """Reshape [M*B, C, D] -> [B, T, D]."""
    # [M*B, C, D] -> [M, B, C, D] -> [B, M, C, D] -> [B, T, D]
    x = x_chunks.view(num_chunks, B, x_chunks.shape[1], x_chunks.shape[2]).transpose(0, 1)
    return x.reshape(B, T, x_chunks.shape[2])


def _expand_states_for_chunks(
    boundary_states: list[torch.Tensor],
    num_chunks: int,
    B: int,
) -> list[torch.Tensor]:
    """Expand boundary states [B, M-1, D] to per-chunk initial states [M*B, ...].

    First chunk uses zeros, subsequent chunks use translated boundaries.
    """
    # boundary_states: list of [B, M-1, D] per layer
    # Returns: list of [M*B, ...] per layer
    expanded = []
    for bs in boundary_states:
        # bs: [B, M-1, D]
        # First chunk: zeros
        zero_state = torch.zeros(B, 1, bs.shape[2], device=bs.device, dtype=bs.dtype)
        # Concatenate: [B, M, D]
        all_boundaries = torch.cat([zero_state, bs], dim=1)  # [B, M, D]
        # Expand to chunks: [B, M, D] -> [M, B, D] -> [M*B, D]
        all_boundaries = all_boundaries.transpose(0, 1).reshape(num_chunks * B, -1)
        expanded.append(all_boundaries)
    return expanded


class ParallelRNNTrainer(nn.Module):
    """Wrapper that enables parallel training of any RNN via chunkwise decomposition.

    During training (per layer):
    - Scaffold scans the layer's input to produce boundary summaries
    - Translator maps summaries to target boundary states
    - Target RNN layer runs in parallel across chunks from boundary states

    During inference:
    - Runs target RNN sequentially (scaffold and translator discarded)
    """

    def __init__(
        self,
        target_rnn: nn.Module,
        chunk_size: int,
        scaffold_dim: int,
        target_type: str = "mlp",
        num_layers: int | None = None,
        hidden_dim: int | None = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        detach_boundary: bool = False,
        use_gdn2_init: bool = True,
    ) -> None:
        super().__init__()
        self.target = target_rnn
        self.chunk_size = chunk_size
        self.scaffold_dim = scaffold_dim
        self.detach_boundary = detach_boundary
        self.use_gdn2_init = use_gdn2_init

        # Infer target properties if not provided
        if num_layers is None:
            num_layers = getattr(target_rnn, "num_layers", 1)
        if hidden_dim is None:
            hidden_dim = getattr(target_rnn, "hidden_dim", scaffold_dim)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Per-layer scaffolds (each scans its layer's input)
        self.scaffolds = nn.ModuleList(
            [
                MinGRUScaffold(
                    input_dim=hidden_dim,  # Each layer's input is hidden_dim
                    hidden_dim=scaffold_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Per-layer translators
        self.translators = nn.ModuleList(
            [
                Translator(
                    input_dim=scaffold_dim,
                    hidden_dim=hidden_dim,
                    target_type=target_type,
                    dropout=dropout,
                    use_gdn2_init=use_gdn2_init,
                )
                for _ in range(num_layers)
            ]
        )

        # Input projection to hidden_dim (if needed) for first layer
        self.input_proj = nn.Identity()
        if hasattr(target_rnn, "input_dim"):
            input_dim = getattr(target_rnn, "input_dim", hidden_dim)
            if isinstance(input_dim, int) and input_dim != hidden_dim:
                self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, state: RNNStateList | None = None) -> torch.Tensor:
        """Forward pass: parallel chunkwise in training, sequential in inference.

        Args:
            x: Input [B, T, D]
            state: Optional initial states

        Returns:
            Logits [B, T, C]
        """
        # In inference mode, run target RNN sequentially (no scaffold/translator)
        if not self.training:
            return self.target(x, state)[0]

        B, T, _ = x.shape
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size

        # Project input to hidden_dim
        x = self.input_proj(x)

        # Process layer by layer
        layer_input = x
        current_state = state

        for layer_idx in range(self.num_layers):
            # Get the RNN and FF sublayers for this layer
            rnn_layer, ff_layer = self._get_layer_pair(layer_idx)

            # Step 1: Scaffold scan on THIS layer's input
            scaffold_states = self.scaffolds[layer_idx](layer_input)  # [B, T, scaffold_dim]

            # Step 2: Get boundary states at chunk boundaries
            boundary_indices = [m * self.chunk_size for m in range(1, num_chunks)]
            if not boundary_indices:
                # Single chunk - run layer sequentially
                layer_input, current_state = self._run_layer_sequential(
                    layer_idx, layer_input, current_state
                )
                continue

            boundary_scaffold = scaffold_states[:, boundary_indices, :]  # [B, M-1, scaffold_dim]

            # Step 3: Translate to target boundary states for this layer
            boundary_states = self.translators[layer_idx](boundary_scaffold)  # [B, M-1, hidden_dim]

            # Optional: detach boundary states to isolate gradient paths
            if self.detach_boundary:
                boundary_states = boundary_states.detach()

            # Step 4: Run RNN sublayer in parallel across chunks
            layer_output = self._run_rnn_layer_parallel(
                layer_idx, rnn_layer, layer_input, boundary_states, current_state
            )

            # Step 5: Run FF sublayer on full sequence (no chunking needed)
            layer_output = ff_layer(layer_output)

            # Prepare for next layer
            layer_input = layer_output

        # Final classifier
        final_norm = getattr(self.target, "final_norm", nn.Identity())
        classifier = getattr(self.target, "classifier", nn.Identity())
        logits = classifier(final_norm(layer_input))
        return logits

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
        layer_output, new_state = rnn_layer(layer_input, state[layer_idx] if state else None)
        layer_output = ff_layer(layer_output)
        return layer_output, new_state

    def _run_rnn_layer_parallel(
        self,
        layer_idx: int,
        rnn_layer: nn.Module,
        layer_input: torch.Tensor,
        boundary_states: torch.Tensor,  # [B, M-1, D]
        initial_state: RNNStateList | None,
    ) -> torch.Tensor:
        """Run RNN layer in parallel across chunks.

        Args:
            rnn_layer: The RNN sublayer (e.g., MultiHeadRNNLayer, MinGRULayer)
            layer_input: [B, T, D]
            boundary_states: [B, M-1, D] translated boundary states
            initial_state: Initial states for the layer

        Returns:
            Layer output [B, T, D]
        """
        B, T, D = layer_input.shape
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size

        # Reshape input for parallel chunk processing: [B, T, D] -> [M*B, C, D]
        chunked_input, num_chunks, C = _reshape_for_chunks(layer_input, self.chunk_size)

        # Expand boundary states to per-chunk initial states: [B, M, D] -> [M*B, D]
        # boundary_states is [B, M-1, D], prepend zeros for first chunk
        zero_boundary = torch.zeros(
            B, 1, self.hidden_dim, device=boundary_states.device, dtype=boundary_states.dtype
        )
        all_boundaries = torch.cat([zero_boundary, boundary_states], dim=1)  # [B, M, D]
        chunk_initial_states = all_boundaries.transpose(0, 1).reshape(
            num_chunks * B, -1
        )  # [M*B, D]

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
            else:
                # Single vector: [B, D] -> expand to [M*B, D]
                h0 = h0.unsqueeze(0).expand(num_chunks, -1, -1).reshape(num_chunks * B, -1)
            # Override first chunk's state with provided initial state
            chunk_initial_states[:B] = h0[:B]

        # Run RNN layer on chunked input
        # The RNN layer's forward expects [B, T, D] and state
        # We give it [M*B, C, D] with expanded states
        chunk_state = RNNState(hidden=chunk_initial_states)
        chunk_output, _ = rnn_layer.forward(chunked_input, chunk_state)  # type: ignore[misc]

        # Reshape back: [M*B, C, D] -> [B, T, D]
        layer_output = _reshape_from_chunks(chunk_output, B, T, num_chunks)

        # Trim padding if any
        return layer_output[:, :T, :]

    def get_boundary_states(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Get translated boundary states for all layers for inspection."""
        x = self.input_proj(x)
        B, T, _ = x.shape
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        boundary_indices = [m * self.chunk_size for m in range(1, num_chunks)]

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

    def compute_boundary_error(
        self,
        x: torch.Tensor,
        true_states: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute boundary reconstruction error ε_m = ||h_true - h_tilde|| / ||h_true||.

        Args:
            x: Input [B, T, D]
            true_states: List of true hidden states [B, T, D] per layer (from sequential run)

        Returns:
            Mean boundary error across layers and boundaries
        """
        x = self.input_proj(x)
        B, T, _ = x.shape
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        boundary_indices = [m * self.chunk_size for m in range(1, num_chunks)]

        if not boundary_indices:
            return torch.tensor(0.0, device=x.device)

        errors = []
        layer_input = x

        for layer_idx in range(self.num_layers):
            scaffold_states = self.scaffolds[layer_idx](layer_input)
            boundary_scaffold = scaffold_states[:, boundary_indices, :]
            predicted_boundaries = self.translators[layer_idx](boundary_scaffold)  # [B, M-1, D]

            # Get true boundary states
            true_boundaries = true_states[layer_idx][:, boundary_indices, :]  # [B, M-1, D]

            # Compute relative error
            diff = predicted_boundaries - true_boundaries
            error = diff.norm(dim=-1) / (true_boundaries.norm(dim=-1) + 1e-8)
            errors.append(error.mean())

            # For next layer inspection, we'd need actual layer output
            # This is approximate since we don't run the layer

        return torch.stack(errors).mean()

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

    def eval(self) -> ParallelRNNTrainer:
        """Set evaluation mode."""
        return self.train(False)

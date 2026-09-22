"""Nano-RNN language model: nanoGPT/nanoRWKV skeleton with an RNN mixer.

Reference mapping (nanoGPT ``model.py``, RWKV-v4 ``src/model.py``):

- Embed: GPT ``wte + wpe``; RWKV ``emb`` only. Nano keeps ``wte`` and an
  optional ``wpe`` (default off: the recurrence carries order, saving
  ``max_seq_len * hidden_dim`` params).
- Mixer slot: GPT ``CausalSelfAttention``, RWKV ``TimeMix`` (WKV). Nano
  puts any RNN-protocol cell here: ``(B, T, D) -> (B, T, D)``, causal by
  construction, with a ``step`` path for O(1) decoding.
- FFN slot: GPT ``MLP``, RWKV ``ChannelMix``. Nano reuses the GPT MLP
  (``FeedForwardSublayer``) by default, optionally ``RationalFeedForward``.
- Tail: ``final_norm`` (ln_f) + ``classifier`` (lm_head), tied to ``wte``
  by default.

Mixer/state contract (all mixers satisfy this)::

    forward(x [B, T, D], state: RNNState) -> (out [B, T, D], RNNState)
    step(x_t [B, D], state: RNNState) -> (out [B, D], RNNState)

State layouts: mlp/rkan/min_gru/min_lstm ``[B, H, Dh]``; m2rnn
``[B, N, K, V]`` plus ``extra["conv_cache"]``.

Residual note: mixer and FF sublayers already contain their own
``x + proj(...)`` residual (see ``mlp_rnn.py``, ``min_gru.py``), so the
block does not add a second outer residual. The structure is still
norm-first + residual-bearing sublayer, as in nanoGPT/nanoRWKV.
"""

from __future__ import annotations

import functools
from typing import Any

import torch
from torch import nn

from .base import BaseRNNModel, RMSNorm, RNNState, RNNStateList
from .m2rnn import DEFAULT_KEY_DIM, DEFAULT_VALUE_DIM, M2RNNLayer
from .min_gru import MinGRULayer
from .min_lstm import MinLSTMLayer
from .mlp_rnn import FeedForwardSublayer, MultiHeadRNNLayer, make_mlp_heads
from .registry import register_model
from .rkan import RationalFeedForward, RKANHead

MIXER_TYPES = ("mlp", "rkan", "min_gru", "min_lstm", "m2rnn")


def _build_mlp_mixer(
    hidden_dim: int, num_heads: int, dropout: float, kw: dict[str, Any]
) -> nn.Module:
    return MultiHeadRNNLayer(
        hidden_dim,
        num_heads,
        head_factory=functools.partial(
            make_mlp_heads,
            mlp_hidden_mult=int(kw.get("mlp_hidden_mult", 4)),
            num_layers=int(kw.get("mlp_num_layers", 2)),
            dropout=dropout,
        ),
        dropout=dropout,
    )


def _build_rkan_mixer(
    hidden_dim: int, num_heads: int, dropout: float, kw: dict[str, Any]
) -> nn.Module:
    return MultiHeadRNNLayer(
        hidden_dim,
        num_heads,
        head_factory=functools.partial(
            RKANHead,
            rkan_degree=int(kw.get("rkan_degree", 3)),
            rkan_alpha=float(kw.get("rkan_alpha", 1.0)),
            rkan_beta=float(kw.get("rkan_beta", 1.0)),
            rkan_iota=float(kw.get("rkan_iota", 1.0)),
            rkan_mapping=str(kw.get("rkan_mapping", "algebraic_infinite")),
            rkan_type=str(kw.get("rkan_type", "jacobi")),
            rkan_num_basis=int(kw.get("rkan_num_basis", 4)),
        ),
        dropout=dropout,
    )


def _build_min_gru_mixer(
    hidden_dim: int, num_heads: int, dropout: float, kw: dict[str, Any]
) -> nn.Module:
    return MinGRULayer(hidden_dim, num_heads, dropout)


def _build_min_lstm_mixer(
    hidden_dim: int, num_heads: int, dropout: float, kw: dict[str, Any]
) -> nn.Module:
    return MinLSTMLayer(hidden_dim, num_heads, dropout)


def _build_m2rnn_mixer(
    hidden_dim: int, num_heads: int, dropout: float, kw: dict[str, Any]
) -> nn.Module:
    return M2RNNLayer(
        hidden_dim,
        num_heads,
        key_dim=int(kw.get("key_dim", DEFAULT_KEY_DIM)),
        value_dim=int(kw.get("value_dim", DEFAULT_VALUE_DIM)),
        dropout=dropout,
    )


MIXER_BUILDERS = {
    "mlp": _build_mlp_mixer,
    "rkan": _build_rkan_mixer,
    "min_gru": _build_min_gru_mixer,
    "min_lstm": _build_min_lstm_mixer,
    "m2rnn": _build_m2rnn_mixer,
}


def build_mixer_layer(
    mixer_type: str,
    hidden_dim: int,
    num_heads: int,
    dropout: float = 0.0,
    **kwargs: Any,
) -> nn.Module:
    """Build one mixer sublayer for the given ``mixer_type``."""
    builder = MIXER_BUILDERS.get(mixer_type)
    if builder is None:
        raise ValueError(f"Unknown mixer_type: {mixer_type}")
    return builder(hidden_dim, num_heads, dropout, kwargs)


def _build_ff_layer(
    ff_type: str,
    hidden_dim: int,
    mlp_hidden_mult: int,
    dropout: float,
    rkan_kwargs: dict[str, Any],
) -> nn.Module:
    if ff_type == "rkan":
        return RationalFeedForward(hidden_dim, dropout=dropout, **rkan_kwargs)
    if ff_type == "mlp":
        return FeedForwardSublayer(hidden_dim, mlp_hidden_mult, dropout)
    raise ValueError(f"Unknown ff_type: {ff_type}")


@register_model("nano_rnn")
class NanoRNN(BaseRNNModel):
    """nanoGPT/nanoRWKV-style decoder with an RNN-protocol mixer.

    Deviation from sibling models: no ``emb_norm`` (nanoGPT has none;
    embeddings feed blocks directly) and the model owns its token
    embedding (``wte``), accepting token IDs or pre-embedded inputs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 12,
        dropout: float = 0.0,
        vocab_size: int = 50257,
        mixer_type: str = "min_gru",
        mlp_hidden_mult: int = 4,
        mlp_num_layers: int = 2,
        ff_type: str = "mlp",
        tie_embeddings: bool = True,
        use_pos_emb: bool = False,
        max_seq_len: int = 1024,
        key_dim: int = DEFAULT_KEY_DIM,
        value_dim: int = DEFAULT_VALUE_DIM,
        rkan_degree: int = 3,
        rkan_alpha: float = 1.0,
        rkan_beta: float = 1.0,
        rkan_iota: float = 1.0,
        rkan_mapping: str = "algebraic_infinite",
        rkan_type: str = "jacobi",
        rkan_num_basis: int = 4,
        **kwargs: object,
    ) -> None:
        super().__init__(input_dim, hidden_dim, num_layers, num_heads, dropout)
        if mixer_type not in MIXER_TYPES:
            raise ValueError(f"Unknown mixer_type: {mixer_type}")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.mixer_type = mixer_type
        self.vocab_size = vocab_size
        self.ff_type = ff_type
        self.tie_embeddings = tie_embeddings
        self.use_pos_emb = use_pos_emb
        self.max_seq_len = max_seq_len
        self.key_dim = key_dim
        self.value_dim = value_dim

        mixer_kwargs: dict[str, Any] = {
            "mlp_hidden_mult": mlp_hidden_mult,
            "mlp_num_layers": mlp_num_layers,
            "key_dim": key_dim,
            "value_dim": value_dim,
            "rkan_degree": rkan_degree,
            "rkan_alpha": rkan_alpha,
            "rkan_beta": rkan_beta,
            "rkan_iota": rkan_iota,
            "rkan_mapping": rkan_mapping,
            "rkan_type": rkan_type,
            "rkan_num_basis": rkan_num_basis,
        }
        rkan_kwargs: dict[str, Any] = {
            "rkan_degree": rkan_degree,
            "rkan_alpha": rkan_alpha,
            "rkan_beta": rkan_beta,
            "rkan_iota": rkan_iota,
            "rkan_mapping": rkan_mapping,
            "rkan_type": rkan_type,
            "rkan_num_basis": rkan_num_basis,
        }

        self.wte = nn.Embedding(vocab_size, hidden_dim)
        self.wpe: nn.Embedding | None = (
            nn.Embedding(max_seq_len, hidden_dim) if use_pos_emb else None
        )
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        build_mixer_layer(
                            mixer_type, hidden_dim, num_heads, dropout, **mixer_kwargs
                        ),
                        _build_ff_layer(ff_type, hidden_dim, mlp_hidden_mult, dropout, rkan_kwargs),
                    ]
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, vocab_size, bias=False)
        if tie_embeddings:
            self.classifier.weight = self.wte.weight

        self._init_weights(num_layers)

    def _init_weights(self, num_layers: int) -> None:
        """GPT-2 init: N(0, 0.02), residual projs scaled by 1/sqrt(2L)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        scale = (2 * max(1, num_layers)) ** -0.5
        for layer_pair in self.layers:
            mixer, ff = layer_pair
            for proj_name in ("output_proj", "W_o"):
                proj = getattr(mixer, proj_name, None)
                if isinstance(proj, nn.Linear):
                    nn.init.normal_(proj.weight, mean=0.0, std=0.02 * scale)
            down = getattr(ff, "down", None)
            if isinstance(down, nn.Linear):
                nn.init.normal_(down.weight, mean=0.0, std=0.02 * scale)

    def resize_vocab(self, new_vocab_size: int) -> None:
        """Resize tied embedding/lm_head, preserving overlapping rows."""
        if new_vocab_size == self.vocab_size:
            return
        old_embed = self.wte.weight.detach()
        new_embed = nn.Embedding(new_vocab_size, self.hidden_dim, device=old_embed.device)
        nn.init.normal_(new_embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            overlap = min(old_embed.shape[0], new_vocab_size)
            new_embed.weight[:overlap] = old_embed[:overlap]
        self.wte = new_embed
        self.classifier = nn.Linear(self.hidden_dim, new_vocab_size, bias=False)
        if self.tie_embeddings:
            self.classifier.weight = self.wte.weight
        self.vocab_size = new_vocab_size

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.long:
            token_emb = self.wte(x)
            if self.wpe is not None:
                positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
                token_emb = token_emb + self.wpe(positions)
            return self.drop(token_emb)
        return x

    def forward(
        self, x: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, RNNStateList]:
        """Forward over IDs [B, T] or embeddings [B, T, D] -> logits [B, T, V]."""
        x = self._embed(x)
        if state is None:
            state = self.init_state(x.shape[0], x.device)

        new_states_list: list[RNNState] = []
        for index, layer_pair in enumerate(self.layers):
            mixer, ff_layer = layer_pair
            x, new_state = mixer(x, state[index])
            new_states_list.append(new_state)
            x = ff_layer(x)

        return self.classifier(self.final_norm(x)), RNNStateList(new_states_list)

    def step(
        self, x_t: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, RNNStateList]:
        """Single-token step: IDs [B] or embedding [B, D] -> logits [B, V]."""
        if x_t.dim() == 1 and x_t.dtype == torch.long:
            x_t = self._embed(x_t.unsqueeze(1)).squeeze(1)
        elif x_t.dim() == 2 and x_t.dtype == torch.long:
            x_t = self._embed(x_t).squeeze(1)
        if state is None:
            state = self.init_state(x_t.shape[0], x_t.device)

        new_states_list: list[RNNState] = []
        for index, layer_pair in enumerate(self.layers):
            mixer, ff_layer = layer_pair
            x_t, new_state = mixer.step(x_t, state[index])
            new_states_list.append(new_state)
            x_t = ff_layer(x_t)

        return self.classifier(self.final_norm(x_t)), RNNStateList(new_states_list)

    def init_state(self, batch_size: int, device: torch.device) -> RNNStateList:
        head_dim = self.hidden_dim // self.num_heads
        states: list[RNNState] = []
        for _ in range(self.num_layers):
            if self.mixer_type == "m2rnn":
                proj_dim = self.num_heads * (self.key_dim + self.key_dim + self.value_dim)
                states.append(
                    RNNState(
                        hidden=torch.zeros(
                            batch_size,
                            self.num_heads,
                            self.key_dim,
                            self.value_dim,
                            device=device,
                        ),
                        extra={"conv_cache": torch.zeros(batch_size, proj_dim, 3, device=device)},
                    )
                )
            elif self.mixer_type in ("min_gru", "min_lstm"):
                states.append(
                    RNNState(hidden=torch.ones(batch_size, self.num_heads, head_dim, device=device))
                )
            else:
                states.append(
                    RNNState(
                        hidden=torch.zeros(batch_size, self.num_heads, head_dim, device=device)
                    )
                )
        return RNNStateList(states)

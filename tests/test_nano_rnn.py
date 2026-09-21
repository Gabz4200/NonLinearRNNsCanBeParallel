"""Tests for the nano-RNN language model scaffold."""

from __future__ import annotations

import pytest
import torch

import nonlinearrnnscanbeparallel.models  # noqa: F401
from nonlinearrnnscanbeparallel.models.nano_rnn import MIXER_TYPES, NanoRNN
from nonlinearrnnscanbeparallel.models.registry import get_model


def _make(mixer_type: str, **kwargs: object) -> NanoRNN:
    defaults: dict[str, object] = {
        "input_dim": 32,
        "hidden_dim": 32,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.0,
        "vocab_size": 64,
        "mixer_type": mixer_type,
    }
    defaults.update(kwargs)
    model = get_model("nano_rnn", **defaults)
    assert isinstance(model, NanoRNN)
    return model


@pytest.mark.parametrize("mixer_type", MIXER_TYPES, ids=MIXER_TYPES)
def test_when_forward_with_ids_then_logits_shape(mixer_type: str) -> None:
    model = _make(mixer_type)
    ids = torch.randint(0, 64, (2, 16))
    logits, states = model(ids)
    assert logits.shape == (2, 16, 64)
    assert len(states) == 2


@pytest.mark.parametrize("mixer_type", MIXER_TYPES, ids=MIXER_TYPES)
def test_when_step_sequence_then_matches_forward(mixer_type: str) -> None:
    model = _make(mixer_type)
    model.eval()
    ids = torch.randint(0, 64, (2, 12))
    logits, _ = model(ids)

    state = None
    steps = []
    for t in range(ids.shape[1]):
        step_logits, state = model.step(ids[:, t], state)
        steps.append(step_logits.unsqueeze(1))
    sequential = torch.cat(steps, dim=1)

    torch.testing.assert_close(sequential, logits, rtol=1e-4, atol=1e-5)


def test_when_tie_embeddings_then_weights_shared() -> None:
    model = _make("min_gru", tie_embeddings=True)
    assert model.classifier.weight is model.wte.weight


def test_when_pos_emb_disabled_then_no_wpe() -> None:
    model = _make("min_gru", use_pos_emb=False)
    assert model.wpe is None


def test_when_pos_emb_enabled_then_wpe_added() -> None:
    model = _make("min_gru", use_pos_emb=True, max_seq_len=32)
    assert model.wpe is not None


def test_when_resize_vocab_then_classifier_matches() -> None:
    model = _make("min_gru", tie_embeddings=True)
    model.resize_vocab(128)
    assert model.vocab_size == 128
    assert model.classifier.weight is model.wte.weight
    logits, _ = model(torch.randint(0, 128, (1, 8)))
    assert logits.shape == (1, 8, 128)


def test_when_preembedded_float_input_then_forward_accepts() -> None:
    model = _make("min_gru")
    embedded = torch.randn(2, 10, 32)
    logits, _ = model(embedded)
    assert logits.shape == (2, 10, 64)

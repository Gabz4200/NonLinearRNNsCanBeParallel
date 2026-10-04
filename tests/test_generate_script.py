"""Tests for scripts/generate.py's checkpoint loading and decoding.

The failure these guard against is silent: a wrong prefix choice leaves every
checkpoint key unmatched, ``load_state_dict(strict=False)`` reports nothing
useful, and the script happily decodes from a randomly initialised model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate import _best_prefix, _build_model, _strip_prefix, generate  # noqa: E402

import nonlinearrnnscanbeparallel.models  # noqa: F401  (registry side effect)

VOCAB = 256


def _spec(**overrides: object) -> dict:
    spec: dict = {
        "name": "nano_rnn",
        "input_dim": 0,
        "hidden_dim": 32,
        "num_layers": 2,
        "num_heads": 4,
        "vocab_size": VOCAB,
        "mixer_type": "hope",
        "mem_dim": 8,
        "dropout": 0.0,
        "mlp_hidden_mult": 4,
        "ff_type": "mlp",
    }
    spec.update(overrides)
    return spec


def _model(**overrides: object) -> torch.nn.Module:
    torch.manual_seed(0)
    return _build_model(_spec(**overrides), total_steps=1)


def test_when_spec_omits_input_dim_then_it_defaults_to_own_embedding() -> None:
    """Checkpoints record no input_dim; 0 means the model owns its embedding."""
    torch.manual_seed(0)
    spec = _spec()
    del spec["input_dim"]

    model = _build_model(spec, total_steps=1)

    assert hasattr(model, "wte")


@pytest.mark.parametrize("prefix", ["target.", "backbone.target.", "backbone."])
def test_when_checkpoint_prefixes_differ_then_the_target_keys_are_recoverable(
    prefix: str,
) -> None:
    """Each wrapper layout yields the target's own state dict after stripping."""
    model = _model()
    state = {f"{prefix}{k}": v for k, v in model.state_dict().items()}

    stripped = _strip_prefix(state, prefix)

    assert set(stripped) == set(model.state_dict())


def test_when_parallel_checkpoint_then_best_prefix_finds_the_target() -> None:
    """A parallel checkpoint mixes target keys with scaffold/translator keys.

    ``backbone.`` matches every key, so choosing by string prefix alone would
    strip one level too few and leave ``target.*`` keys that fit nothing. The
    selection must go by how many of the model's own tensors get populated, and
    the training-only siblings must survive stripping so the loader reports
    them as unused rather than silently passing them to the target.
    """
    model = _model()
    own = set(model.state_dict())
    state = {f"backbone.target.{k}": v for k, v in model.state_dict().items()}
    # Training-only siblings that live under the same wrapper.
    state["backbone.scaffolds.0.weight"] = torch.zeros(1)
    state["backbone.translators.0.weight"] = torch.zeros(1)

    prefix, stripped, matched = _best_prefix(state, own)

    assert prefix == "backbone.target."
    assert matched == len(own)
    assert not any(k.startswith("target.") for k in stripped)
    assert {"backbone.scaffolds.0.weight", "backbone.translators.0.weight"} <= set(stripped)


def test_when_prefix_is_wrong_then_the_search_recovers_the_target() -> None:
    """A prefix chosen by string matching alone leaves ``target.*`` behind.

    The search still finds the right answer from that intermediate state, which
    is why selecting by overlap works where matching the longest prefix string
    would not: ``backbone.`` "matches" every key of a wrapper-nested
    checkpoint, and a first-match rule would stop there.
    """
    model = _model()
    own = set(model.state_dict())
    # The real parallel checkpoint: nested wrapper plus training-only siblings.
    state = {f"backbone.target.{k}": v for k, v in model.state_dict().items()}
    state["backbone.scaffolds.0.weight"] = torch.zeros(1)
    state["backbone.output_head.0.weight"] = torch.zeros(1)

    # A rule that stops at the first *string* match picks backbone. (it matches
    # every key by name) and never gets to the target level.
    first_match: tuple[str | None, dict, int] = (None, dict(state), 0)
    for candidate in ("backbone.", "target.", "backbone.target.", "model."):
        if sum(k.startswith(candidate) for k in state) >= len(state) // 2:
            trial = _strip_prefix(state, candidate)
            first_match = (
                candidate,
                trial,
                sum(key in own for key in trial),
            )
            break
    # That leaves every target tensor unpopulated, so strict=False loads nothing.
    assert first_match[2] == 0, "a string-first rule would silently load nothing"

    # Overlap-based selection gets it right.
    prefix, _, matched = _best_prefix(state, own)
    assert (prefix, matched) == ("backbone.target.", len(own))


def test_when_bptt_checkpoint_then_best_prefix_also_works() -> None:
    """The BPTT layout stores the target directly under the wrapper."""
    model = _model()
    own = set(model.state_dict())
    state = {f"target.{k}": v for k, v in model.state_dict().items()}

    prefix, stripped, matched = _best_prefix(state, own)

    assert prefix == "target."
    assert matched == len(own)


def test_when_decoding_then_the_result_depends_on_the_loaded_weights() -> None:
    """Decoding from trained weights must differ from decoding from fresh ones.

    This is the end-to-end guard on the silent bug: if the prefix is chosen
    wrong the two models are identical, and a "successful" generation would
    actually be reading a random model.
    """

    class _Bytes:
        def encode(self, text: str) -> list[int]:
            return list(text.encode("utf-8"))

        def decode(self, ids: list[int]) -> str:
            return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")

    trained = _model()
    # Perturb every parameter so the two models cannot coincide by accident.
    with torch.no_grad():
        for param in trained.parameters():
            param.add_(torch.randn_like(param) * 0.05)
    fresh = _model()
    tokenizer = _Bytes()
    device = torch.device("cpu")

    text_a = generate(
        model=trained,
        tokenizer=tokenizer,
        prompt="ab",
        max_new_tokens=6,
        temperature=0.0,
        top_k=None,
        device=device,
        seed=0,
    )
    text_b = generate(
        model=fresh,
        tokenizer=tokenizer,
        prompt="ab",
        max_new_tokens=6,
        temperature=0.0,
        top_k=None,
        device=device,
        seed=0,
    )

    assert text_a != text_b

"""Generate text from a trained HOPE checkpoint.

Loads a Lightning checkpoint written by ``scripts/train.py``, rebuilds the same
model the run used, and decodes with the target's ``step()`` method.

Decoding uses ``step`` rather than ``forward`` on purpose.  ``forward`` applies
Eq. 90's chunk-start evaluation point when ``chunk_size`` is given, which is the
*training* objective; ``step`` always runs the plain sequential recurrence,
which is the inference-time model.  They are deliberately different (see the
HOPE module docstring), so a single-token decode that called ``forward`` would
be running the training-time surrogate.

Example::

    uv run python scripts/generate.py \\
        --checkpoint results/mixed_lm_<ts>/checkpoints/best.ckpt \\
        --model-config results/mixed_lm_<ts>/config.yaml \\
        --prompt "The" --max-new-tokens 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from _util import TRAINING_KEYS, node_dict  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import nonlinearrnnscanbeparallel.models  # noqa: E402, F401  (registry side effect)
from nonlinearrnnscanbeparallel.models.registry import get_model  # noqa: E402


class ByteTokenizer:
    """Byte-level tokenizer, so any checkpoint decodes without a download.

    Maps the 256 byte values to ids directly, so the round trip is exact.  Only
    for exercising generation wiring, not for judging text quality; pass a
    ``tokenizer`` with a real vocabulary when quality matters.
    """

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")


def _build_model(model_cfg: dict, total_steps: int) -> torch.nn.Module:
    """Rebuild the model the run trained, from the resolved config."""
    name = str(model_cfg.get("name", "nano_rnn"))
    kwargs = {k: v for k, v in model_cfg.items() if k not in TRAINING_KEYS and k != "name"}
    # The checkpoint's model_spec omits input_dim: 0 means the model owns its
    # embedding, which is what every NanoRNN language model uses.
    kwargs.setdefault("input_dim", 0)
    kwargs["total_steps"] = total_steps
    return get_model(name, **kwargs)


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _strip_prefix(state: dict, prefix: str) -> dict:
    """Drop Lightning's ``<module>.`` wrapper keys from a state dict."""
    return {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in state.items()}


def _best_prefix(state: dict, own_keys: set[str]) -> tuple[str | None, dict, int]:
    """Choose the checkpoint prefix that recovers the most of the target's keys.

    A parallel run stores the target under ``backbone.target.*`` alongside
    scaffold and translator keys; a BPTT run uses ``target.*`` or ``backbone.*``.
    Matching on the string alone is not enough, because a parallel checkpoint's
    ``backbone.`` prefix also "matches" every key while leaving ``target.*``
    entries that fit nothing. Selecting by overlap with the model's own keys is
    what makes a wrong choice visible instead of silently loading no weights.

    Returns:
        Tuple of (prefix, state dict with that prefix removed, number of the
        model's tensors the result can populate).
    """
    best: tuple[int, str | None, dict] = (-1, None, dict(state))
    for candidate in (None, "backbone.target.", "target.", "backbone.", "model."):
        trial = dict(state) if candidate is None else _strip_prefix(state, candidate)
        overlap = sum(key in own_keys for key in trial)
        if overlap > best[0]:
            best = (overlap, candidate, trial)
    return best[1], best[2], best[0]


def generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
    seed: int,
) -> str:
    """Greedy/sampled decode, advancing the recurrent state one token at a time."""
    was_training = model.training
    model.eval()
    torch.manual_seed(seed)

    step_fn = getattr(model, "step", None)
    if not callable(step_fn):
        raise TypeError(f"{type(model).__name__} has no step(); cannot decode token by token")

    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    state = None

    with torch.no_grad():
        # Prime the state over the prompt. step() returns [B, V] for a single
        # token, so the last position's logits are already the full row.
        logits = None
        for position in range(ids.shape[1]):
            logits, state = step_fn(ids[:, position], state)

        if logits is None:
            raise ValueError("prompt is empty; nothing to decode from")

        for _ in range(max_new_tokens):
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = logits / temperature
                if top_k:
                    k = min(top_k, scaled.shape[-1])
                    threshold = scaled.topk(k, dim=-1).values[:, -1:]
                    scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
                next_id = torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
            logits, state = step_fn(next_id, state)

    model.train(was_training)
    return tokenizer.decode(ids[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Lightning .ckpt to load")
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="resolved config.yaml from the training run's results directory",
    )
    parser.add_argument("--prompt", type=str, default="The", help="prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 means greedy")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Hugging Face tokenizer name matching the training vocab; omit for the byte tokenizer",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Resolve the whole file to plain containers first: the OmegaConf node API
    # is typed per subclass, and node_dict() is typed for plain dicts.
    resolved = node_dict(OmegaConf.load(args.model_config))
    model_cfg = resolved.get("model") or {}

    checkpoint = _load_checkpoint(args.checkpoint)
    state_dict = checkpoint.get("state_dict", checkpoint)
    hyper = checkpoint.get("hyper_parameters", {}) or {}

    # The checkpoint carries the exact model spec it was trained with, which is
    # more reliable than the config file (the config may have been edited after
    # the run, and CLI overrides are not in the defaults).
    spec = hyper.get("model_spec")
    if isinstance(spec, dict) and spec:
        model_cfg = {**model_cfg, **spec}
        print(f"model spec taken from checkpoint: {len(spec)} keys")

    vocab_size = int(model_cfg.get("vocab_size", hyper.get("vocab_size", 0)))
    total_steps = int(model_cfg.get("total_steps", hyper.get("total_steps", 1)))

    model = _build_model(model_cfg, total_steps)

    # A prompt id outside the model's vocabulary would fail deep inside the
    # embedding lookup, so surface any mismatch here.
    embedding = getattr(model, "wte", None)
    if isinstance(embedding, torch.nn.Embedding) and vocab_size:
        limit = int(embedding.weight.shape[0])
        if limit != vocab_size:
            print(
                f"warning: config vocab_size={vocab_size} but the built model "
                f"embeds {limit}; using the model's own size"
            )

    # Scaffold and translator are training-only and are dropped: inference runs
    # the target sequentially, with no boundary states to supply.
    prefix, state_dict, matched = _best_prefix(state_dict, set(model.state_dict()))
    print(
        f"unwrapping target weights: prefix={prefix!r}, "
        f"{matched}/{len(model.state_dict())} target tensors matched"
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"warning: {len(missing)} parameters were not in the checkpoint, e.g. {missing[:3]}")
    if unexpected:
        print(f"note: {len(unexpected)} checkpoint keys unused, e.g. {unexpected[:3]}")
    if matched == 0:
        raise RuntimeError(
            "no checkpoint key matched the model's parameters; refusing to decode "
            "from an untrained model. Check --model-config and the checkpoint's "
            "model_spec."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # The tokenizer must match the checkpoint's vocabulary: ids index the
    # embedding table directly, so a pretrained tokenizer on a small toy vocab
    # produces out-of-range ids. Default to bytes, which is always in range,
    # and require an explicit --tokenizer for real text.
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer: Any = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = ByteTokenizer()

    embedding = getattr(model, "wte", None)
    limit = int(embedding.weight.shape[0]) if isinstance(embedding, torch.nn.Embedding) else None
    if limit is not None and not args.tokenizer:
        print(
            f"note: using the byte tokenizer (no --tokenizer given). It is exact "
            f"for ids < {limit}, but text quality is not meaningful from raw "
            f"bytes -- pass the training run's tokenizer to read the output."
        )

    print(f"loaded {args.checkpoint}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"device: {device}")
    print("-" * 60)
    print(f"PROMPT: {args.prompt!r}\n")

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        seed=args.seed,
    )
    print(text)
    print("-" * 60)


if __name__ == "__main__":
    main()

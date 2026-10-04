"""Shared batch collation for the causal-LM data modules."""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def collate_lm_samples(
    samples: list[dict[str, torch.Tensor]], pad_id: int
) -> dict[str, torch.Tensor]:
    """Right-pad a batch of causal-LM samples to the longest one and stack.

    Every causal-LM source emits ``{input_ids, labels, attention_mask}`` and
    pads with its own tokenizer's ``pad_id``, so the sources differ only in
    which id that is. Labels pad with ``IGNORE_INDEX``, which is what the
    cross-entropy losses skip.

    Args:
        samples: per-item dicts with ``input_ids``, ``labels``, ``attention_mask``.
        pad_id: the padding token id for ``input_ids``.

    Returns:
        Batched dict with the same three keys.
    """
    max_len = max(sample["input_ids"].shape[0] for sample in samples)
    input_ids, labels, attention_mask = [], [], []
    for sample in samples:
        pad_len = max_len - sample["input_ids"].shape[0]
        input_ids.append(
            torch.cat([sample["input_ids"], sample["input_ids"].new_full((pad_len,), pad_id)])
        )
        labels.append(
            torch.cat([sample["labels"], sample["labels"].new_full((pad_len,), IGNORE_INDEX)])
        )
        attention_mask.append(
            torch.cat([sample["attention_mask"], sample["attention_mask"].new_zeros(pad_len)])
        )
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }

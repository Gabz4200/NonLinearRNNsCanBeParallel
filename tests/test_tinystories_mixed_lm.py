"""Tests for TinyStoriesInstruct splitting and the mixed-LM interleave (no network)."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from nonlinearrnnscanbeparallel.data import tinystories_instruct
from nonlinearrnnscanbeparallel.data.mixed_lm import MixedLMDataset
from nonlinearrnnscanbeparallel.data.tinystories_instruct import (
    TinyStoriesInstructConfig,
    TinyStoriesInstructDataset,
    split_stories,
)


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 32

    def __call__(self, text: str, truncation: bool, max_length: int, return_tensors: Any) -> dict:
        ids = [(ord(c) % (self.vocab_size - 2)) + 2 for c in text][:max_length]
        return {"input_ids": ids}


_SAMPLE_ROW = (
    "Dialogue\nSummary: A and B play.\nStory: A and B play happily.\n<|endoftext|>"
    "Summary: C steals.\nStory: C steals a bike.\n<|endoftext|>\n\n"
)


class _StubDataset:
    def __init__(self, items: list[int]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> int:
        return self._items[index]


def test_when_row_split_on_endtext_then_stories() -> None:
    stories = split_stories(_SAMPLE_ROW, "<|endoftext|>")
    assert len(stories) == 2
    assert "A and B play happily" in stories[0]
    assert "C steals a bike" in stories[1]


def test_when_row_only_delimiters_then_no_stories() -> None:
    assert split_stories("<|endoftext|><|endoftext|>", "<|endoftext|>") == []


def test_when_dataset_then_indexed_and_masked() -> None:
    cfg = TinyStoriesInstructConfig(block_size=16)
    dataset = TinyStoriesInstructDataset(
        split_stories(_SAMPLE_ROW, "<|endoftext|>"), cast(Any, _FakeTokenizer()), cfg
    )
    assert len(dataset) == 2
    sample = dataset[1]
    assert sample["input_ids"].dtype == torch.long
    assert sample["input_ids"].shape[0] <= 16
    assert sample["labels"].shape == sample["input_ids"].shape
    assert (-100 in sample["labels"]) is False  # no padding inside one story


def test_when_mixed_then_batches_alternate_round_robin() -> None:
    mixed = MixedLMDataset([_StubDataset([0, 1, 2]), _StubDataset([10, 11, 12])])
    assert len(mixed) == 6
    assert [mixed[i] for i in range(6)] == [0, 10, 1, 11, 2, 12]


def test_when_mixed_source_shorter_then_balanced() -> None:
    mixed = MixedLMDataset([_StubDataset([0, 1, 2, 3]), _StubDataset([10])])
    assert len(mixed) == 2
    assert [mixed[i] for i in range(2)] == [0, 10]


def test_when_source_empty_then_errors() -> None:
    with pytest.raises(ValueError):
        MixedLMDataset([_StubDataset([1, 2]), _StubDataset([])])


def test_when_tiny_module_collate_then_padded_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        tinystories_instruct.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *a, **k: tokenizer),
    )
    dm = tinystories_instruct.TinyStoriesInstructDataModule(
        {"subset_num_samples": 2, "block_size": 16, "batch_size": 2, "num_workers": 0}
    )
    object.__setattr__(dm, "tokenizer", tokenizer)
    cfg = TinyStoriesInstructConfig(block_size=16)
    dataset = TinyStoriesInstructDataset(
        ["short", "a much much longer story padded to exercise the collate"],
        cast(Any, tokenizer),
        cfg,
    )
    batch = dm._collate([dataset[0], dataset[1]])
    expect_cols = max(dataset[0]["input_ids"].shape[0], dataset[1]["input_ids"].shape[0])
    assert batch["input_ids"].shape[1] == expect_cols
    assert batch["labels"].shape == batch["input_ids"].shape
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()


def test_when_top_level_omits_shared_keys_then_source_defaults_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nonlinearrnnscanbeparallel.data import mixed_lm

    captured: dict[str, Any] = {}

    class _FakeSource:
        def __init__(self, data_config: dict[str, Any]) -> None:
            captured.update(data_config)
            self.train_dataset = _StubDataset([1, 2])
            self.val_dataset = _StubDataset([1])
            self.test_dataset = _StubDataset([1])
            self.tokenizer = _FakeTokenizer()

        def setup(self, stage: str | None = None) -> None:
            pass

    monkeypatch.setattr(mixed_lm, "_SOURCE_FACTORIES", {"wikipedia": _FakeSource})
    module = mixed_lm.MixedLMDataModule(
        {"batch_size": 2, "sources": [{"name": "wikipedia", "articles_per_subset": 4}]}
    )
    module.setup("fit")
    assert "tokenizer_name" not in captured
    assert "block_size" not in captured
    assert captured["articles_per_subset"] == 4

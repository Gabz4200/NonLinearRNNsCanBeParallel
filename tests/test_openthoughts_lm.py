"""Tests for the OpenThoughts LM data module (no network access)."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from nonlinearrnnscanbeparallel.data import openthoughts_lm
from nonlinearrnnscanbeparallel.data.openthoughts_lm import (
    OpenThoughtsLMConfig,
    OpenThoughtsLMDataModule,
    OpenThoughtsLMDataset,
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


def _example(i: int) -> dict[str, str]:
    return {"question": f"q{i}", "reasoning": f"r{i}", "answer_or_no_answer": f"a{i}"}


def test_when_dataset_getitem_then_labels_and_mask() -> None:
    cfg = OpenThoughtsLMConfig(max_seq_len=16)
    dataset = OpenThoughtsLMDataset([_example(0)], _FakeTokenizer(), cfg)
    sample = dataset[0]
    assert sample["input_ids"].shape == sample["labels"].shape == sample["attention_mask"].shape
    assert sample["input_ids"].dtype == torch.long


def test_when_collate_pads_then_labels_masked() -> None:
    cfg = OpenThoughtsLMConfig(max_seq_len=16, batch_size=2)
    dataset = OpenThoughtsLMDataset([_example(0), _example(1)], _FakeTokenizer(), cfg)
    dm = OpenThoughtsLMDataModule.__new__(OpenThoughtsLMDataModule)
    dm.config = cfg
    object.__setattr__(dm, "tokenizer", _FakeTokenizer())
    batch = dm._collate([dataset[0], dataset[1]])
    assert batch["input_ids"].shape[0] == 2
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()


def test_when_setup_streams_slice_then_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = [_example(i) for i in range(20)]
    monkeypatch.setattr(openthoughts_lm, "load_dataset", lambda *a, **k: samples)
    monkeypatch.setattr(
        openthoughts_lm.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *a, **k: _FakeTokenizer()),
    )
    dm = OpenThoughtsLMDataModule(
        {"subset_num_samples": 20, "max_seq_len": 16, "batch_size": 2, "num_workers": 0}
    )
    dm.setup("fit")
    assert dm.tokenizer_vocab_size == 32
    train, val, test = dm.train_dataset, dm.val_dataset, dm.test_dataset
    assert train is not None and val is not None and test is not None
    assert len(cast(Any, train)) + len(cast(Any, val)) + len(cast(Any, test)) == 20
    assert len(cast(Any, train)) > 0
    batch = next(iter(dm.train_dataloader()))
    assert set(batch.keys()) == {"input_ids", "labels", "attention_mask"}

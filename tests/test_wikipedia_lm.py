"""Tests for the Wikipedia LM data module (no network access)."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from nonlinearrnnscanbeparallel.data import wikipedia_lm
from nonlinearrnnscanbeparallel.data.wikipedia_lm import (
    WikipediaLMConfig,
    WikipediaLMDataModule,
    WikipediaLMDataset,
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


def _article(i: int, words: int = 100) -> dict[str, str]:
    return {
        "id": str(i),
        "url": f"https://en.wikipedia.org/wiki/A{i}",
        "title": f"A{i}",
        "text": ("wxyz " * words).strip(),
    }


def test_when_article_blocked_then_fixed_size_blocks() -> None:
    cfg = WikipediaLMConfig(block_size=16)
    dm = WikipediaLMDataModule.__new__(WikipediaLMDataModule)
    dm.config = cfg
    object.__setattr__(dm, "tokenizer", _FakeTokenizer())
    blocks = dm._block_article(_article(0))
    assert len(blocks) > 1
    assert all(len(b) <= 16 for b in blocks)
    dataset = WikipediaLMDataset(blocks, cast(Any, _FakeTokenizer()), cfg)
    sample = dataset[0]
    assert sample["input_ids"].shape == (16,)
    assert sample["input_ids"].dtype == torch.long
    assert (sample["labels"][sample["attention_mask"] == 0] == -100).all()


def test_when_setup_streams_subsets_then_article_level_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    per_subset = [[_article(i), _article(10 + i)] for i in range(2)]

    def _fake_load(name: str, subset: str, **kwargs: Any) -> list[dict[str, str]]:
        idx = ["20231101.en", "20231101.pt"].index(subset)
        return per_subset[idx]

    monkeypatch.setattr(wikipedia_lm, "load_dataset", _fake_load)
    monkeypatch.setattr(
        wikipedia_lm.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *a, **k: _FakeTokenizer()),
    )
    dm = WikipediaLMDataModule(
        {
            "subsets": ["20231101.en", "20231101.pt"],
            "articles_per_subset": 2,
            "block_size": 16,
            "max_article_tokens": 512,
            "batch_size": 2,
            "num_workers": 0,
        }
    )
    dm.setup("fit")
    assert dm.tokenizer_vocab_size == 32
    train, val, test = dm.val_dataset, dm.val_dataset, dm.test_dataset
    train = dm.train_dataset
    assert train is not None and val is not None and test is not None
    n_blocks = len(cast(Any, train)) + len(cast(Any, val)) + len(cast(Any, test))
    assert n_blocks > 4  # 4 articles cut into 16-token blocks
    batch = next(iter(dm.train_dataloader()))
    assert set(batch.keys()) == {"input_ids", "labels", "attention_mask"}
    assert batch["input_ids"].shape[1] == 16

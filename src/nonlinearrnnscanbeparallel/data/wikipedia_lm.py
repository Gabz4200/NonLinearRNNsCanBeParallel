"""Wikimedia Wikipedia causal LM data module (English + Portuguese subsets).

Loads configurable slices of ``wikimedia/wikipedia`` via streaming
(``itertools.islice``), so the full dump is never downloaded. The HF schema is
``{"id", "url", "title", "text"}``; each article is rendered as
``"<title>\\n\\n<text>"``. The tokenizer is GPT-2 BPE (``gpt2``,
vocab 50257) with ``pad_token = eos_token``. Batch contract stays
``{input_ids [B, T], labels [B, T]}`` matching every other data module;
loss masking uses ``labels == -100`` on padded positions.

Language note: Wikipedia ships one subset per language/edition, so there is
no separate Brazilian-Portuguese edition — ``20231101.pt`` is the single
Portuguese edition and includes Brazilian content. Configure ``subsets``
with ``"<date>.<lang>"`` codes, e.g. ``["20231101.en", "20231101.pt"]``.

Articles are long, so each one is tokenized once and cut into ``block_size``
blocks (the trailing short block is padded and masked). Splitting into
train/val/test happens at the article level, before blocking, so blocks from
one article never leak across splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from typing import Any, cast

import lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class WikipediaLMConfig:
    """Configuration for :class:`WikipediaLMDataModule`."""

    dataset_name: str = "wikimedia/wikipedia"
    subsets: list[str] = field(default_factory=lambda: ["20231101.en", "20231101.pt"])
    dataset_split: str = "train"
    articles_per_subset: int = 1000
    subset_seed: int = 42
    tokenizer_name: str = "gpt2"
    max_seq_len: int = 1024
    block_size: int = 1024
    max_article_tokens: int = 8192
    val_split: float = 0.02
    test_split: float = 0.02
    batch_size: int = 8
    num_workers: int = 0
    streaming: bool = True
    trust_remote_code: bool = False


class WikipediaLMDataset(Dataset):
    """Fixed-length token blocks with ``-100``-masked padding."""

    def __init__(
        self,
        blocks: list[list[int]],
        tokenizer: PreTrainedTokenizerBase,
        config: WikipediaLMConfig,
    ) -> None:
        self.blocks = blocks
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = self.blocks[index]
        pad_len = self.config.block_size - len(ids)
        input_ids = torch.tensor(ids + [self.tokenizer.pad_token_id] * pad_len, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        attention_mask = torch.cat(
            [torch.ones(len(ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class WikipediaLMDataModule(pl.LightningDataModule):
    """Streams Wikipedia slices, blocks articles, and splits by article."""

    def __init__(self, data_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = WikipediaLMConfig(**data_config or {})
        if config.max_seq_len != config.block_size:
            config.max_seq_len = config.block_size
        self.config = config
        self.tokenizer = cast(
            PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(config.tokenizer_name)
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    @property
    def tokenizer_vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    def _load_articles(self) -> list[dict[str, Any]]:
        cfg = self.config
        articles: list[dict[str, Any]] = []
        for subset in cfg.subsets:
            ds = load_dataset(
                cfg.dataset_name,
                subset,
                split=cfg.dataset_split,
                streaming=cfg.streaming,
                trust_remote_code=cfg.trust_remote_code,
            )
            articles.extend(cast(list[dict[str, Any]], list(islice(ds, cfg.articles_per_subset))))
        return articles

    def _block_article(self, article: dict[str, Any]) -> list[list[int]]:
        cfg = self.config
        text = f"{article.get('title', '')}\n\n{article.get('text', '')}"
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_article_tokens,
            return_tensors=None,
        )
        ids = list(enc["input_ids"])
        if len(ids) < 2:
            ids = ids + [self.tokenizer.eos_token_id]
        return [ids[i : i + cfg.block_size] for i in range(0, len(ids), cfg.block_size)]

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return
        cfg = self.config
        articles = self._load_articles()
        from numpy.random import default_rng

        rng = default_rng(cfg.subset_seed)
        indices = rng.permutation(len(articles))
        n_total = len(indices)
        n_test = int(n_total * cfg.test_split)
        n_val = int(n_total * cfg.val_split)

        test_idx = set(indices[:n_test].tolist())
        val_idx = set(indices[n_test : n_test + n_val].tolist())
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        for i, article in enumerate(articles):
            if i in test_idx:
                splits["test"].append(article)
            elif i in val_idx:
                splits["val"].append(article)
            else:
                splits["train"].append(article)

        blocked = {
            name: [block for article in split for block in self._block_article(article)]
            for name, split in splits.items()
        }
        self.train_dataset = WikipediaLMDataset(blocked["train"], self.tokenizer, cfg)
        self.val_dataset = WikipediaLMDataset(blocked["val"], self.tokenizer, cfg)
        self.test_dataset = WikipediaLMDataset(blocked["test"], self.tokenizer, cfg)

    def _collate(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(s["input_ids"].shape[0] for s in samples)
        input_ids, labels, attention_mask = [], [], []
        for s in samples:
            pad = max_len - s["input_ids"].shape[0]
            input_ids.append(
                torch.cat(
                    [s["input_ids"], s["input_ids"].new_full((pad,), self.tokenizer.pad_token_id)]
                )
            )
            labels.append(torch.cat([s["labels"], s["labels"].new_full((pad,), -100)]))
            attention_mask.append(
                torch.cat([s["attention_mask"], s["attention_mask"].new_zeros(pad)])
            )
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }

    def _dataloader(self, dataset: Dataset) -> DataLoader:
        cfg = self.config
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=(dataset is self.train_dataset),
            num_workers=cfg.num_workers,
            collate_fn=self._collate,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return self._dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return self._dataloader(self.test_dataset)

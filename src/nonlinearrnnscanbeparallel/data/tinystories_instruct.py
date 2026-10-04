"""roneneldan/TinyStoriesInstruct causal LM data module.

Rows are multi-story blobs: several stories separated by the literal end-of
text token ``<|endoftext|>`` (each segment carries its own metadata header,
e.g. ``Dialogue\\nWords: ...\\nSummary: ...\\nFeatures: ...\\nStory: ...``).
We split each row on that token and treat every non-empty segment as one
training item, so the model sees story text + its structured header.

Streaming (``itertools.islice``) keeps the download small. The tokenizer is
GPT-2 BPE (``gpt2``, vocab 50257) with ``pad_token = eos_token``. Batch
contract stays ``{input_ids [B, T], labels [B, T]}`` matching every other
data module; loss masking uses ``labels == -100`` on padded positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any, cast

import lightning as pl
import torch
from datasets import load_dataset
from numpy.random import default_rng
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .collate import collate_lm_samples


@dataclass
class TinyStoriesInstructConfig:
    """Configuration for :class:`TinyStoriesInstructDataModule`."""

    dataset_name: str = "roneneldan/TinyStoriesInstruct"
    dataset_split: str = "train"
    subset_num_samples: int = 32  # rows; each row yields several stories
    subset_seed: int = 42
    text_key: str = "text"
    end_of_text_token: str = "<|endoftext|>"
    tokenizer_name: str = "gpt2"
    max_seq_len: int = 256
    block_size: int = 256
    val_split: float = 0.02
    test_split: float = 0.02
    batch_size: int = 8
    num_workers: int = 0
    streaming: bool = True
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.val_split < 1.0:
            raise ValueError("val_split must be in (0.0, 1.0)")
        if not 0.0 < self.test_split < 1.0:
            raise ValueError("test_split must be in (0.0, 1.0)")


def split_stories(row_text: str, end_of_text_token: str) -> list[str]:
    """Split one TinyStories row into non-empty story segments."""
    parts = [p.strip() for p in row_text.split(end_of_text_token)]
    return [p for p in parts if p]


class TinyStoriesInstructDataset(Dataset):
    """One item per story segment (row split on end-of-text token)."""

    def __init__(
        self,
        stories: list[str],
        tokenizer: PreTrainedTokenizerBase,
        config: TinyStoriesInstructConfig,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.ids: list[list[int]] = [
            list(
                tokenizer(
                    story,
                    truncation=True,
                    max_length=config.block_size,
                    return_tensors=None,
                )["input_ids"]
            )
            for story in stories
        ]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = self.ids[index]
        if len(ids) < 2:
            ids = ids + [self.tokenizer.eos_token_id]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }


class TinyStoriesInstructDataModule(pl.LightningDataModule):
    """Streams TinyStories rows, splits them into stories, and splits."""

    def __init__(self, data_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = TinyStoriesInstructConfig(**data_config or {})
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

    def _load_rows(self) -> list[str]:
        cfg = self.config
        ds = load_dataset(
            cfg.dataset_name,
            split=cfg.dataset_split,
            streaming=True,
            trust_remote_code=cfg.trust_remote_code,
        )
        return [
            str(example.get(cfg.text_key, "")) for example in islice(ds, cfg.subset_num_samples)
        ]

    def _all_stories(self, rows: list[str]) -> list[str]:
        stories: list[str] = []
        for row in rows:
            stories.extend(split_stories(row, self.config.end_of_text_token))
        return stories

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return
        cfg = self.config
        stories = self._all_stories(self._load_rows())
        rng = default_rng(cfg.subset_seed)
        indices = rng.permutation(len(stories))
        # floor(frac * n) gives 0 for small slices, so guarantee at least one
        # val and one test item; raise only when no training items would remain.
        n_test = max(1, int(len(stories) * cfg.test_split))
        n_val = max(1, int(len(stories) * cfg.val_split))
        if n_test + n_val >= len(stories):
            raise ValueError(
                f"not enough stories ({len(stories)}) for a nonempty train/val/test split; "
                "raise subset_num_samples"
            )

        test_idx = set(indices[:n_test].tolist())
        val_idx = set(indices[n_test : n_test + n_val].tolist())
        train_items = [s for i, s in enumerate(stories) if i not in test_idx and i not in val_idx]
        val_items = [stories[i] for i in indices[n_test : n_test + n_val]]
        test_items = [stories[i] for i in indices[:n_test]]

        self.train_dataset = TinyStoriesInstructDataset(train_items, self.tokenizer, cfg)
        self.val_dataset = TinyStoriesInstructDataset(val_items, self.tokenizer, cfg)
        self.test_dataset = TinyStoriesInstructDataset(test_items, self.tokenizer, cfg)

    def _collate(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return collate_lm_samples(samples, self.tokenizer.pad_token_id)

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

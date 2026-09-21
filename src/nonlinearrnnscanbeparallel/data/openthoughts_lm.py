"""OpenThoughts-114k causal LM data module (subset only).

Loads a configurable slice of ``open-thoughts/OpenThoughts-114k`` via
streaming (``itertools.islice``), so the full dataset is never downloaded.
The HF schema is ``{"system": str, "conversations": [{"from", "value"}]}``;
conversation turns are rendered as ``"<from>: <value>"`` joined by blank
lines, prefixed by the system prompt. The tokenizer is GPT-2 BPE
(``gpt2``, vocab 50257) with ``pad_token = eos_token``. Batch contract
stays ``{input_ids [B, T], labels [B, T]}`` matching every other data
module; loss masking uses ``labels == -100`` on padded positions.
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

FALLBACK_KEYS = ["question", "reasoning"]
ANSWER_KEYS = ["answer", "answer_or_no_answer"]


@dataclass
class OpenThoughtsLMConfig:
    """Configuration for :class:`OpenThoughtsLMDataModule`."""

    dataset_name: str = "open-thoughts/OpenThoughts-114k"
    dataset_split: str = "train"
    subset_num_samples: int = 5000
    subset_seed: int = 42
    tokenizer_name: str = "gpt2"
    max_seq_len: int = 1024
    block_size: int = 1024
    val_split: float = 0.02
    test_split: float = 0.02
    batch_size: int = 8
    num_workers: int = 4
    streaming: bool = True
    trust_remote_code: bool = False
    num_proc: int = 4
    text_field_template: str = "{question}\n\n{reasoning}\n\n{answer}"
    # OpenThoughts-114k schema.
    system_key: str = "system"
    conversations_key: str = "conversations"
    turn_key_from: str = "from"
    turn_key_value: str = "value"
    # Generic fallback keys for other instruct datasets.
    question_key: str = "question"
    reasoning_key: str = "reasoning"
    answer_key: str = "answer_or_no_answer"
    extra_separator: str = "\n\n"
    extra_keys: list[str] = field(default_factory=list)


class OpenThoughtsLMDataset(Dataset):
    """Tokenized fixed-length blocks over the sliced HF dataset."""

    def __init__(self, hf_dataset, tokenizer, config: OpenThoughtsLMConfig) -> None:
        self.examples = list(hf_dataset)
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self) -> int:
        return len(self.examples)

    def _render_text(self, example) -> str:
        cfg = self.config
        if cfg.conversations_key in example and example[cfg.conversations_key]:
            parts: list[str] = []
            system = example.get(cfg.system_key)
            if system:
                parts.append(str(system))
            for turn in example[cfg.conversations_key]:
                parts.append(f"{turn[cfg.turn_key_from]}: {turn[cfg.turn_key_value]}")
            return cfg.extra_separator.join(parts)
        try:
            text = cfg.text_field_template.format(
                question=example[cfg.question_key],
                reasoning=example[cfg.reasoning_key],
                answer=example[cfg.answer_key],
            )
        except KeyError:
            parts = []
            for key in FALLBACK_KEYS:
                if key in example:
                    parts.append(str(example[key]))
            text = cfg.extra_separator.join(parts)
        for key in cfg.extra_keys:
            if key in example:
                text = text + cfg.extra_separator + str(example[key])
        return text

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        text = self._render_text(self.examples[index])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_seq_len,
            return_tensors=None,
        )
        ids = list(enc["input_ids"])
        if len(ids) < 2:
            ids = list(enc["input_ids"]) + [self.tokenizer.eos_token_id]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class OpenThoughtsLMDataModule(pl.LightningDataModule):
    """Streams a slice of OpenThoughts-114k, tokenizes, and splits."""

    def __init__(self, data_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = OpenThoughtsLMConfig(**data_config or {})
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

    def _load_slice(self):
        cfg = self.config
        ds = load_dataset(
            cfg.dataset_name,
            split=cfg.dataset_split,
            streaming=True,
            trust_remote_code=cfg.trust_remote_code,
        )
        return islice(ds, cfg.subset_num_samples)

    def setup(self, stage: str | None = None) -> None:
        cfg = self.config
        slice_iter = self._load_slice()
        from numpy.random import default_rng

        rng = default_rng(cfg.subset_seed)
        sample_list = list(slice_iter)
        indices = rng.permutation(len(sample_list))
        n_total = len(indices)
        n_test = int(n_total * cfg.test_split)
        n_val = int(n_total * cfg.val_split)

        test_idx = set(indices[:n_test].tolist())
        val_idx = set(indices[n_test : n_test + n_val].tolist())
        train_items = [
            item for i, item in enumerate(sample_list) if i not in test_idx and i not in val_idx
        ]
        val_items = [sample_list[i] for i in indices[n_test : n_test + n_val]]
        test_items = [sample_list[i] for i in indices[:n_test]]

        self.train_dataset = OpenThoughtsLMDataset(train_items, self.tokenizer, cfg)
        self.val_dataset = OpenThoughtsLMDataset(val_items, self.tokenizer, cfg)
        self.test_dataset = OpenThoughtsLMDataset(test_items, self.tokenizer, cfg)

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

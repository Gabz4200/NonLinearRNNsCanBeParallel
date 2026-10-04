"""Mixed-corpus causal LM data module: alternate two sources per sample.

Combines two causal-LM data sources (e.g. Wikipedia + TinyStoriesInstruct)
into one balanced stream. Samples are interleaved round-robin at the item
level, so consecutive batch slots alternate between the sources (deterministic
alternation, no re-shuffling). Each source keeps its own tokenizer/blocking
via the existing per-source data modules; all sources must share a tokenizer
(block sizes are aligned too).

Batch contract stays ``{input_ids [B, T], labels [B, T]}`` with ``-100``
masking on padding, matching every other module.
"""

from __future__ import annotations

from typing import Any

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .collate import collate_lm_samples
from .tinystories_instruct import TinyStoriesInstructDataModule
from .wikipedia_lm import WikipediaLMDataModule

_SOURCE_FACTORIES: dict[str, type[pl.LightningDataModule]] = {
    "wikipedia": WikipediaLMDataModule,
    "tinystories_instruct": TinyStoriesInstructDataModule,
}

_SHARED_KEYS = (
    "tokenizer_name",
    "block_size",
    "max_seq_len",
    "val_split",
    "test_split",
    "streaming",
    "trust_remote_code",
)


class MixedLMDataset(Dataset):
    """Round-robin interleave of source datasets.

    ``__getitem__(i)`` reads source ``i % n`` at index ``i // n``, so batch
    slots alternate sources in strict order. All sources contribute equally
    (the stream length is ``n * min(len)``).
    """

    def __init__(self, sources: list[Any]) -> None:
        self.sources = sources
        self.n = len(sources)
        self.per_source = min(s.__len__() for s in sources)
        if self.per_source < 1:
            raise ValueError("every mixed source must have at least one item")

    def __len__(self) -> int:
        return self.per_source * self.n

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.sources[index % self.n][index // self.n]


class MixedLMDataModule(pl.LightningDataModule):
    """Balanced alternating blend of ``sources`` (dicts keyed by name)."""

    def __init__(self, data_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = dict(data_config or {})
        self._batch_size = cfg.get("batch_size", 8)
        self._num_workers = cfg.get("num_workers", 0)
        self._shared = {key: value for key in _SHARED_KEYS if (value := cfg.get(key)) is not None}

        self._sources: list[tuple[str, dict[str, Any]]] = []
        for source in cfg.get("sources", []):
            name = source.get("name")
            if name is None:
                raise ValueError("each mixed source requires a 'name' key")
            self._sources.append((str(name), dict(source)))
        if not self._sources:
            raise ValueError("mixed_lm requires data.sources (list of source configs)")

        self._pad_id = 0
        self._modules: list[Any] = []
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    @property
    def tokenizer_vocab_size(self) -> int:
        return int(getattr(self._modules[0], "tokenizer_vocab_size", 50257))

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return
        for name, source_cfg in self._sources:
            factory = _SOURCE_FACTORIES.get(name)
            if factory is None:
                raise ValueError(f"Unknown mixed source: {name}")
            base = {k: v for k, v in source_cfg.items() if k != "name"}
            for key, value in self._shared.items():
                base.setdefault(key, value)
            module = factory(base)
            module.setup("fit")
            self._modules.append(module)

        self._pad_id = int(getattr(self._modules[0].tokenizer, "pad_token_id", 0))
        self.train_dataset = MixedLMDataset([m.train_dataset for m in self._modules])
        self.val_dataset = MixedLMDataset([m.val_dataset for m in self._modules])
        self.test_dataset = MixedLMDataset([m.test_dataset for m in self._modules])

    def _collate(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return collate_lm_samples(samples, self._pad_id)

    def _dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,  # round-robin alternation must stay in order
            num_workers=self._num_workers,
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

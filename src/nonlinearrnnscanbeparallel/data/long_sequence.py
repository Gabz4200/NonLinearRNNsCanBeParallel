"""Long sequence tasks for testing parallel chunkwise RNN training.

Provides tasks that scale to very long sequences (16k-256k tokens):
- Copy task: remember and reproduce a sequence after a delay
- Induction task: pattern completion in long sequences
- Language modeling on long sequences
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class LongSequenceSample:
    """Sample for long sequence tasks."""

    input_ids: torch.Tensor  # [seq_len]
    labels: torch.Tensor  # [seq_len] - target tokens
    task_type: str  # "copy", "induction", "lm"


class LongSequenceCopyDataset(Dataset):
    """
    Copy task: remember a random sequence and reproduce it after a delay.

    This tests long-range memory and is a classic RNN benchmark.
    Sequence: [BOS] + random_sequence + [SEP] + delay_tokens + [CUE] + target_sequence
    """

    def __init__(
        self,
        num_samples: int = 1000,
        vocab_size: int = 128,
        seq_len: int = 16384,  # 16k tokens
        delay_len: int = 100,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.delay_len = delay_len
        self.seed = seed

        # Special tokens
        self.bos_token = 0
        self.sep_token = 1
        self.cue_token = 2
        self.pad_token = 3
        self.first_content_token = 4
        self.last_content_token = vocab_size - 1

        # Validate
        content_len = seq_len - 4 - delay_len  # BOS, SEP, CUE, delay, EOS
        if content_len <= 0:
            raise ValueError(f"seq_len too small for delay_len")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> LongSequenceSample:
        rng = self._rng_for_index(index)

        # Content length (what we need to copy)
        content_len = (self.seq_len - 4 - self.delay_len) // 2

        # Generate random content to memorize
        content = rng.integers(
            self.first_content_token, self.last_content_token + 1, size=content_len, dtype=np.int64
        )

        # Build sequence: BOS + content + SEP + delay + CUE + target + EOS
        delay_tokens = np.full(self.delay_len, self.pad_token, dtype=np.int64)

        seq = np.concatenate(
            [
                [self.bos_token],
                content,
                [self.sep_token],
                delay_tokens,
                [self.cue_token],
                content,
                [self.vocab_size - 1],  # EOS token
            ]
        )

        seq_len = len(seq)
        assert seq_len <= self.seq_len, f"Sequence too long: {seq_len} > {self.seq_len}"

        input_ids = torch.zeros(self.seq_len, dtype=torch.long)
        input_ids[:seq_len] = torch.tensor(seq, dtype=torch.long)

        # Labels: only predict the copied content (after CUE)
        labels = torch.full((self.seq_len,), -100, dtype=torch.long)
        # Target starts after BOS + content + SEP + delay + CUE
        target_start = 1 + len(content) + 1 + self.delay_len + 1
        target_end = target_start + len(content)
        labels[target_start:target_end] = torch.tensor(content, dtype=torch.long)

        return LongSequenceSample(input_ids=input_ids, labels=labels, task_type="copy")

    def _rng_for_index(self, index: int) -> np.random.Generator:
        ss = np.random.SeedSequence(self.seed + index)
        return np.random.default_rng(ss)

    @staticmethod
    def collate_fn(samples: list[LongSequenceSample]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([s.input_ids for s in samples]),
            "labels": torch.stack([s.labels for s in samples]),
        }


class LongSequenceInductionDataset(Dataset):
    """
    Induction task: pattern completion in long sequences.

    Tests the ability to learn abstract patterns and apply them later.
    Sequence: [BOS] + A + B + A + [MASK] where model must predict B.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        vocab_size: int = 128,
        seq_len: int = 16384,
        pattern_len: int = 10,
        num_patterns: int = 5,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.pattern_len = pattern_len
        self.num_patterns = num_patterns
        self.seed = seed

        # Special tokens
        self.bos_token = 0
        self.mask_token = 1
        self.pad_token = 2
        self.first_content_token = 3
        self.last_content_token = vocab_size - 2

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> LongSequenceSample:
        rng = self._rng_for_index(index)

        # Generate patterns: list of (A, B) pairs
        patterns = []
        for _ in range(self.num_patterns):
            A = rng.integers(
                self.first_content_token,
                self.last_content_token + 1,
                size=self.pattern_len,
                dtype=np.int64,
            )
            B = rng.integers(
                self.first_content_token,
                self.last_content_token + 1,
                size=self.pattern_len,
                dtype=np.int64,
            )
            patterns.append((A, B))

        # Build sequence: BOS + A1+B1 + A2+B2 + ... + An + [MASK] + Bn
        seq = [self.bos_token]
        for A, B in patterns[:-1]:
            seq.extend(A)
            seq.extend(B)

        # Last pattern: A + MASK
        A_last, B_last = patterns[-1]
        seq.extend(A_last)
        seq.append(self.mask_token)

        seq_len = len(seq)
        assert seq_len <= self.seq_len, f"Sequence too long: {seq_len} > {self.seq_len}"

        input_ids = torch.zeros(self.seq_len, dtype=torch.long)
        input_ids[:seq_len] = torch.tensor(seq, dtype=torch.long)

        # Labels: predict B at mask position
        labels = torch.full((self.seq_len,), -100, dtype=torch.long)
        labels[seq_len - 1] = B_last[0]  # Predict first token of B

        return LongSequenceSample(input_ids=input_ids, labels=labels, task_type="induction")

    def _rng_for_index(self, index: int) -> np.random.Generator:
        ss = np.random.SeedSequence(self.seed + index)
        return np.random.default_rng(ss)

    @staticmethod
    def collate_fn(samples: list[LongSequenceSample]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([s.input_ids for s in samples]),
            "labels": torch.stack([s.labels for s in samples]),
        }


class LongSequenceLMDataModule(pl.LightningDataModule):
    """DataModule for long sequence language modeling tasks."""

    def __init__(
        self,
        data_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg = data_config or {}
        self.task_type = cfg.get("task_type", "copy")
        self.num_samples = cfg.get("num_samples", 1000)
        self.vocab_size = cfg.get("vocab_size", 128)
        self.seq_len = cfg.get("seq_len", 16384)
        self.delay_len = cfg.get("delay_len", 100)
        self.pattern_len = cfg.get("pattern_len", 10)
        self.num_patterns = cfg.get("num_patterns", 5)
        self.seed = cfg.get("seed", 42)
        self.val_split = cfg.get("val_split", 0.1)
        self.test_split = cfg.get("test_split", 0.1)
        self.batch_size = cfg.get("batch_size", 4)  # Small batch for long sequences
        self.num_workers = cfg.get("num_workers", 2)

    def setup(self, stage: str | None = None) -> None:
        if self.task_type == "copy":
            full_dataset = LongSequenceCopyDataset(
                num_samples=self.num_samples,
                vocab_size=self.vocab_size,
                seq_len=self.seq_len,
                delay_len=self.delay_len,
                seed=self.seed,
            )
        elif self.task_type == "induction":
            full_dataset = LongSequenceInductionDataset(
                num_samples=self.num_samples,
                vocab_size=self.vocab_size,
                seq_len=self.seq_len,
                pattern_len=self.pattern_len,
                num_patterns=self.num_patterns,
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        val_size = int(self.num_samples * self.val_split)
        test_size = int(self.num_samples * self.test_split)
        train_size = self.num_samples - val_size - test_size

        split_gen = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=split_gen
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
        )

    @staticmethod
    def _collate_fn(samples: list) -> dict[str, torch.Tensor]:
        # Use the collate_fn from the dataset
        if hasattr(samples[0], "__class__"):
            # Find the appropriate collate_fn
            for cls in type(samples[0]).__mro__:
                if hasattr(cls, "collate_fn") and callable(cls.collate_fn):
                    return cls.collate_fn(samples)
        return {
            "input_ids": torch.stack([s.input_ids for s in samples]),
            "labels": torch.stack([s.labels for s in samples]),
        }

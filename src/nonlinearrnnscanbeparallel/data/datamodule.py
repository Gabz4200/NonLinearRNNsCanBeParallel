"""DataModule for sorted deterministic graph connectivity task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightning
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class GraphConnectivitySample:
    """Single graph connectivity sample."""

    input_ids: torch.Tensor  # [seq_len] - unary encoded sequence
    labels: torch.Tensor  # [seq_len] - binary classification at last token
    source: int
    target: int
    num_nodes: int


def _rng_for_index(seed: int, idx: int) -> np.random.Generator:
    """Deterministic per-index RNG (Definition 11 encoding is fixed given seed+idx).

    Each __getitem__(idx) must return the same sample regardless of access order,
    multiprocessing workers, or random_split composition.
    """
    ss = np.random.SeedSequence(seed + idx)
    return np.random.default_rng(ss)


class SortedGraphConnectivityDataset(Dataset):
    """
    Dataset for sorted deterministic graph connectivity (Definition 11, arXiv:2603.03612).

    Nodes are in topological order i1 < i2 < ... < in, each node has at most one outgoing edge.
    Encoded in unary as specified in the paper: BOS, edges in unary, source, target, EOS.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        max_nodes: int = 32,
        max_seq_len: int = 66,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.max_nodes = max_nodes
        self.max_seq_len = max_seq_len
        self.seed = seed

        # Longest valid sequence: BOS plus edges plus source, target, and EOS.
        # Total is 2*N + 2 tokens, so max_seq_len must cover it.
        min_seq_len = 2 * max_nodes + 2
        if max_seq_len < min_seq_len:
            raise ValueError(
                f"max_seq_len={max_seq_len} must be >= 2*max_nodes+2={min_seq_len} "
                "to avoid truncating EOS tokens."
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> GraphConnectivitySample:
        rng = _rng_for_index(self.seed, index)

        num_nodes = int(rng.integers(2, self.max_nodes + 1))

        # Create edges with at most one outgoing edge per node.
        edges: list[tuple[int, int]] = []
        next_node = [-1] * num_nodes
        for i in range(num_nodes - 1):
            if rng.random() < 0.5:
                j = int(rng.integers(i + 1, num_nodes))
                edges.append((i, j))
                next_node[i] = j

        source = int(rng.integers(0, num_nodes))
        target = int(rng.integers(0, num_nodes))

        def has_path(s: int, t: int) -> bool:
            visited = set()
            curr = s
            while curr != -1 and curr not in visited:
                if curr == t:
                    return True
                visited.add(curr)
                curr = next_node[curr]
            return False

        label = 1 if has_path(source, target) else 0

        # Encode in unary: BOS token, then edges, then source, target, EOS.
        # Integer codes run from 0 for BOS through num_nodes+3 for EOS.
        bos = 0
        source_token = int(num_nodes) + 1
        target_token = int(num_nodes) + 2
        eos = int(num_nodes) + 3

        seq = [bos]
        for i, j in edges:
            seq.append(i + 1)
            seq.append(j + 1)
        seq.append(source_token)
        seq.append(target_token)
        seq.append(eos)

        seq_len = len(seq)
        assert seq_len <= self.max_seq_len, (
            f"Generated sequence length {seq_len} exceeds cap {self.max_seq_len}. "
            "This is a logic error in graph generation."
        )

        input_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
        input_ids[:seq_len] = torch.tensor(seq, dtype=torch.long)

        # Labels: only predict at the last token (EOS position).
        labels = torch.full((self.max_seq_len,), -100, dtype=torch.long)
        labels[seq_len - 1] = label

        return GraphConnectivitySample(
            input_ids=input_ids,
            labels=labels,
            source=source,
            target=target,
            num_nodes=int(num_nodes),
        )

    @staticmethod
    def collate_fn(samples: list[GraphConnectivitySample]) -> dict[str, torch.Tensor]:
        """Collate list of samples into batched tensors."""
        return {
            "input_ids": torch.stack([s.input_ids for s in samples]),
            "labels": torch.stack([s.labels for s in samples]),
        }


class GraphConnectivityDataModule(lightning.LightningDataModule):
    """DataModule for sorted deterministic graph connectivity."""

    def __init__(
        self,
        data_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg = data_config or {}
        self.num_samples = cfg.get("num_samples", 10000)
        self.max_nodes = cfg.get("max_nodes", 32)
        self.max_seq_len = cfg.get("max_seq_len", 66)
        self.seed = cfg.get("seed", 42)
        self.val_split = cfg.get("val_split", 0.1)
        self.test_split = cfg.get("test_split", 0.1)
        self.batch_size = cfg.get("batch_size", 32)
        self.num_workers = cfg.get("num_workers", 4)

    def setup(self, stage: str | None = None) -> None:
        full_dataset = SortedGraphConnectivityDataset(
            num_samples=self.num_samples,
            max_nodes=self.max_nodes,
            max_seq_len=self.max_seq_len,
            seed=self.seed,
        )

        val_size = int(self.num_samples * self.val_split)
        test_size = int(self.num_samples * self.test_split)
        train_size = self.num_samples - val_size - test_size

        # Deterministic splits via a dedicated generator.
        split_gen = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=split_gen
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=SortedGraphConnectivityDataset.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=SortedGraphConnectivityDataset.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=SortedGraphConnectivityDataset.collate_fn,
            num_workers=self.num_workers,
        )

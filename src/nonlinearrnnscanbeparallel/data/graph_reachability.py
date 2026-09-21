"""Graph Reachability (USTCON) Dataset for L-complete problem training.

Encodes undirected graph reachability (s-t connectivity) as sequences for RNN training.
This is an L-complete problem under logspace reductions.

Token vocabulary:
- Token 0: PAD
- Token 1: BOS (beginning of sequence)
- Token 2: EOS (end of sequence)
- Token 3: SEP (separator between graph and query)
- Token 4: NODE token prefix
- Token 5: EDGE token prefix
- Token 6: QUERY token prefix
- Token 7: SOURCE marker
- Token 8: TARGET marker
- Token 9: ANSWER marker (position where model should predict reachability)
- Token 10+: Node IDs (offset by vocab_offset)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

# Shared token constants for cross-module consistency
PAD = 0
BOS = 1
EOS = 2
SEP = 3
NODE_PREFIX = 4
EDGE_PREFIX = 5
QUERY_PREFIX = 6
SOURCE_MARKER = 7
TARGET_MARKER = 8
ANSWER_MARKER = 9  # Position where model predicts 0/1

SPLIT_SEED_OFFSETS = {"train": 0, "val": 100000, "test": 200000}


@dataclass
class GraphReachabilityConfig:
    """Configuration for graph reachability dataset."""

    num_samples: int = 10000
    min_nodes: int = 8
    max_nodes: int = 16
    edge_prob: float = 0.3
    vocab_size: int = 64
    max_seq_len: int = 1024
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 4

    def __post_init__(self) -> None:
        constraints: list[tuple[bool, str]] = [
            (self.num_samples <= 0, "num_samples must be positive"),
            (self.min_nodes < 1, "min_nodes must be >= 1"),
            (self.max_nodes < self.min_nodes, "max_nodes must be >= min_nodes"),
            (not (0.0 <= self.edge_prob <= 1.0), "edge_prob must be in [0.0, 1.0]"),
            (
                self.max_seq_len < 9 + 2 * self.min_nodes,
                f"max_seq_len ({self.max_seq_len}) must be >= "
                f"9 + 2*min_nodes ({9 + 2 * self.min_nodes}) to fit the shortest legal sequence",
            ),
            (self.num_workers < 0, "num_workers must be non-negative"),
            (
                self.vocab_size <= 10 + self.max_nodes - 1,
                f"vocab_size ({self.vocab_size}) must be > "
                f"10 + max_nodes - 1 ({10 + self.max_nodes - 1})",
            ),
            (
                self.val_split is not None and not (0.0 < self.val_split < 1.0),
                "val_split must be in (0.0, 1.0) or None",
            ),
            (
                self.test_split is not None and not (0.0 < self.test_split < 1.0),
                "test_split must be in (0.0, 1.0) or None",
            ),
        ]
        for failed, message in constraints:
            if failed:
                raise ValueError(message)


def generate_undirected_graph(
    num_nodes: int, edge_prob: float, seed: int | None = None
) -> torch.Tensor:
    """Generate random undirected graph as adjacency matrix.

    Args:
        num_nodes: Number of nodes
        edge_prob: Probability of each edge
        seed: Random seed for reproducibility

    Returns:
        Adjacency matrix [num_nodes, num_nodes] (symmetric, no self-loops)
    """
    if seed is not None:
        torch.manual_seed(seed)

    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if torch.rand(1).item() < edge_prob:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj


def compute_reachability(adj: torch.Tensor) -> torch.Tensor:
    """Compute reachability matrix using Floyd-Warshall (transitive closure).

    Args:
        adj: Adjacency matrix [n, n]

    Returns:
        Reachability matrix [n, n] where reach[i, j] = 1 if j reachable from i
    """
    n = adj.shape[0]
    reach = adj.clone().to(torch.bool)
    # Add self-loops for transitive closure
    for i in range(n):
        reach[i, i] = True

    # Floyd-Warshall for transitive closure
    for k in range(n):
        for i in range(n):
            if reach[i, k]:
                reach[i] = reach[i] | reach[k]

    return reach.to(torch.long)


def graph_to_sequence(
    adj: torch.Tensor, source: int, target: int, vocab_offset: int = 10
) -> tuple[torch.Tensor, int]:
    """Encode graph + query as a sequence.

    Encoding scheme:
    - Token 0: PAD
    - Token 1: BOS (beginning of sequence)
    - Token 2: EOS (end of sequence)
    - Token 3: SEP (separator between graph and query)
    - Token 4: NODE token prefix
    - Token 5: EDGE token prefix
    - Token 6: QUERY token prefix
    - Token 7: SOURCE marker
    - Token 8: TARGET marker
    - Token 9: ANSWER marker (position where model should predict reachability)
    - Token 10+: Node IDs (offset by vocab_offset)

    Sequence format:
    [BOS] NODE_0 ... NODE_{n-1} EDGE_0_1 ... EDGE_{n-2}_{n-1}
    SEP QUERY SOURCE_{s} TARGET_{t} ANSWER EOS

    The ANSWER token (9) is where the model should predict reachability (0 or 1).

    Args:
        adj: Adjacency matrix [n, n]
        source: Source node index
        target: Target node index
        vocab_offset: Offset for node ID tokens

    Returns:
        Sequence tensor and label (1 if reachable, 0 otherwise)
    """
    n = adj.shape[0]

    seq = [BOS]

    # Add node tokens
    for i in range(n):
        seq.append(NODE_PREFIX)
        seq.append(vocab_offset + i)

    # Add edge tokens (only upper triangle for undirected)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                seq.append(EDGE_PREFIX)
                seq.append(vocab_offset + i)
                seq.append(vocab_offset + j)

    # Separator
    seq.append(SEP)

    # Query
    seq.append(QUERY_PREFIX)
    seq.append(SOURCE_MARKER)
    seq.append(vocab_offset + source)
    seq.append(TARGET_MARKER)
    seq.append(vocab_offset + target)

    # Answer position - model should predict 0 or 1 here
    seq.append(ANSWER_MARKER)

    seq.append(EOS)

    # Compute label
    reach = compute_reachability(adj)
    label = 1 if reach[source, target] == 1 else 0

    return torch.tensor(seq, dtype=torch.long), label


class GraphReachabilityDataset(Dataset):
    """Dataset for undirected graph reachability (USTCON)."""

    def __init__(
        self,
        config: GraphReachabilityConfig,
        split: str = "train",
    ) -> None:
        self.config = config
        self.split = split

        # Set seed for reproducibility
        base_seed = config.seed
        if split == "val":
            base_seed += 10000
        elif split == "test":
            base_seed += 20000

        random.seed(base_seed)
        torch.manual_seed(base_seed)

        self.samples = []
        self._generate_samples()

    def _generate_samples(self) -> None:
        """Generate balanced samples without changing encoded graph semantics."""
        num_samples = self.config.num_samples
        if self.split == "val":
            num_samples = max(1, int(self.config.num_samples * self.config.val_split))
        elif self.split == "test":
            num_samples = max(1, int(self.config.num_samples * self.config.test_split))

        target_reachable = num_samples // 2
        target_unreachable = num_samples - target_reachable
        reachable_count = 0
        unreachable_count = 0
        attempts = 0
        max_attempts = num_samples * 100

        while (
            reachable_count < target_reachable or unreachable_count < target_unreachable
        ) and attempts < max_attempts:
            sample_seed = self.config.seed + attempts + SPLIT_SEED_OFFSETS.get(self.split, 0)
            torch.manual_seed(sample_seed)
            random.seed(sample_seed)

            num_nodes = random.randint(self.config.min_nodes, self.config.max_nodes)
            adj = generate_undirected_graph(num_nodes, self.config.edge_prob)
            source = random.randint(0, num_nodes - 1)
            target = random.randint(0, num_nodes - 1)
            seq, label = graph_to_sequence(adj, source, target)

            if len(seq) > self.config.max_seq_len:
                attempts += 1
                continue

            if label == 1 and reachable_count >= target_reachable:
                attempts += 1
                continue
            if label == 0 and unreachable_count >= target_unreachable:
                attempts += 1
                continue

            seq = torch.cat(
                [seq, torch.zeros(self.config.max_seq_len - len(seq), dtype=torch.long)]
            )
            if (seq[:-1] == ANSWER_MARKER).sum() != 1:
                raise ValueError("encoded sequence must contain exactly one ANSWER token")

            self.samples.append((seq, label))
            if label == 1:
                reachable_count += 1
            else:
                unreachable_count += 1
            attempts += 1

        if reachable_count < target_reachable or unreachable_count < target_unreachable:
            raise ValueError(
                f"could not generate {num_samples} balanced samples for {self.split!r} "
                f"after {attempts} attempts; accepted reachable={reachable_count}, "
                f"unreachable={unreachable_count}. Increase max_seq_len or reduce "
                f"max_nodes/edge_prob."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        seq, label = self.samples[index]

        # Create labels: -100 everywhere except at ANSWER_MARKER position (9)
        # where the label should be the reachability label (0 or 1)
        labels = torch.full_like(seq[:-1], -100)
        answer_positions = seq[:-1] == 9  # ANSWER_MARKER = 9 in input_ids
        labels[answer_positions] = label

        return {
            "input_ids": seq[:-1].clone(),
            "labels": labels,
            "reachability_label": torch.tensor(label, dtype=torch.long),
        }


class GraphReachabilityDataModule(pl.LightningDataModule):
    """Lightning DataModule for graph reachability."""

    def __init__(self, config: GraphReachabilityConfig) -> None:
        super().__init__()
        self.config = config
        self.train_dataset: GraphReachabilityDataset | None = None
        self.val_dataset: GraphReachabilityDataset | None = None
        self.test_dataset: GraphReachabilityDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = GraphReachabilityDataset(self.config, "train")
            self.val_dataset = GraphReachabilityDataset(self.config, "val")
        if stage == "test" or stage is None:
            self.test_dataset = GraphReachabilityDataset(self.config, "test")

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )


def create_graph_reachability_datamodule(
    num_samples: int = 5000,
    min_nodes: int = 8,
    max_nodes: int = 16,
    edge_prob: float = 0.3,
    vocab_size: int = 128,
    max_seq_len: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> GraphReachabilityDataModule:
    """Factory function to create GraphReachabilityDataModule."""
    config = GraphReachabilityConfig(
        num_samples=num_samples,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        edge_prob=edge_prob,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    return GraphReachabilityDataModule(config)

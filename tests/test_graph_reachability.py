"""Tests for GraphReachabilityDataset invariants."""

from __future__ import annotations

import pytest

from nonlinearrnnscanbeparallel.data.graph_reachability import (
    GraphReachabilityConfig,
    GraphReachabilityDataset,
)


def test_dataset_answer_token_present() -> None:
    """Every sample must contain exactly one ANSWER token (9) in input_ids."""
    config = GraphReachabilityConfig(
        num_samples=50,
        min_nodes=8,
        max_nodes=12,
        edge_prob=0.3,
        max_seq_len=256,
        seed=42,
    )
    ds = GraphReachabilityDataset(config, "train")

    for i in range(len(ds)):
        item = ds[i]
        answer_mask = item["input_ids"] == 9
        assert answer_mask.sum().item() == 1, (
            f"Sample {i} has {answer_mask.sum().item()} answer tokens"
        )


def test_dataset_correct_length() -> None:
    """Every sample's input_ids must have length max_seq_len - 1."""
    config = GraphReachabilityConfig(
        num_samples=50,
        min_nodes=8,
        max_nodes=12,
        edge_prob=0.2,
        max_seq_len=256,
        seed=42,
    )
    ds = GraphReachabilityDataset(config, "train")

    expected_len = config.max_seq_len - 1
    for i in range(len(ds)):
        item = ds[i]
        assert len(item["input_ids"]) == expected_len, (
            f"Sample {i} has length {len(item['input_ids'])}, expected {expected_len}"
        )


def test_dataset_reachability_label_matches_answer() -> None:
    """The reachability_label must match the label at the ANSWER position in labels."""
    config = GraphReachabilityConfig(
        num_samples=50,
        min_nodes=8,
        max_nodes=12,
        edge_prob=0.3,
        max_seq_len=128,
        seed=42,
    )
    ds = GraphReachabilityDataset(config, "train")

    for i in range(len(ds)):
        item = ds[i]
        answer_pos = (item["input_ids"] == 9).nonzero(as_tuple=True)[0]
        assert len(answer_pos) == 1
        pos = answer_pos[0].item()
        # labels has -100 everywhere except at answer position
        assert item["labels"][pos] == item["reachability_label"].item()


def test_dataset_balanced_classes() -> None:
    """Dataset should generate balanced reachable/unreachable samples."""
    config = GraphReachabilityConfig(
        num_samples=100,
        min_nodes=8,
        max_nodes=12,
        edge_prob=0.3,
        max_seq_len=256,
        seed=42,
    )
    ds = GraphReachabilityDataset(config, "train")

    labels = [item["reachability_label"].item() for i in range(len(ds)) for item in [ds[i]]]
    reachable = sum(labels)
    unreachable = len(labels) - reachable
    # Should be roughly balanced (within 1)
    assert abs(reachable - unreachable) <= 1


def test_when_production_like_config_then_balanced_dataset_is_generated() -> None:
    config = GraphReachabilityConfig(
        num_samples=100,
        min_nodes=10,
        max_nodes=20,
        edge_prob=0.3,
        max_seq_len=256,
        seed=42,
    )

    dataset = GraphReachabilityDataset(config, "train")

    labels = [dataset[index]["reachability_label"].item() for index in range(len(dataset))]
    assert len(labels) == 100
    assert sum(labels) == 50


def test_when_all_candidates_are_overlong_generation_fails_finite() -> None:
    config = GraphReachabilityConfig(
        num_samples=10,
        min_nodes=10,
        max_nodes=10,
        edge_prob=1.0,
        max_seq_len=100,
        seed=42,
    )

    with pytest.raises(ValueError, match="after 1000 attempts"):
        GraphReachabilityDataset(config, "train")


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"num_samples": 0}, "num_samples"),
        ({"min_nodes": 0}, "min_nodes"),
        ({"min_nodes": 6, "max_nodes": 5}, "max_nodes"),
        ({"edge_prob": -0.1}, "edge_prob"),
        ({"min_nodes": 5, "max_seq_len": 10}, "max_seq_len"),
        ({"num_workers": -1}, "num_workers"),
        ({"max_nodes": 10, "vocab_size": 15}, "vocab_size"),
        ({"val_split": 0.0}, "val_split"),
        ({"test_split": 1.0}, "test_split"),
    ],
)
def test_invalid_dataset_config_fails_fast(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GraphReachabilityConfig(**updates)



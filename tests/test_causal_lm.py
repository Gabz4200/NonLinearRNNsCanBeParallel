"""Tests for the causal LM loss and metrics."""

from __future__ import annotations

import math

import pytest
import torch

from nonlinearrnnscanbeparallel.logging.metrics import (
    negative_log_likelihood,
    perplexity,
    token_accuracy,
)
from nonlinearrnnscanbeparallel.losses.language_modeling import causal_lm_loss


def test_when_shift_applied_then_matches_manual_cross_entropy() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    labels = torch.randint(0, 7, (2, 5))
    got = causal_lm_loss(logits, labels)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 7), labels[:, 1:].reshape(-1)
    )
    torch.testing.assert_close(got, expected)


def test_when_ignore_index_then_masked_positions_excluded() -> None:
    logits = torch.randn(1, 4, 5)
    labels = torch.tensor([[-100, 1, -100, 2]])
    got = causal_lm_loss(logits, labels)
    # Shifted targets are labels[1:] = [1, -100, 2]; only positions 0 and 2 contribute.
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 5), labels[:, 1:].reshape(-1), ignore_index=-100
    )
    torch.testing.assert_close(got, expected)


def test_when_all_targets_masked_then_nan() -> None:
    logits = torch.randn(1, 3, 4)
    labels = torch.full((1, 3), -100)
    loss = causal_lm_loss(logits, labels)
    assert math.isnan(loss.item())


def test_when_perplexity_computed_then_exp_of_loss() -> None:
    assert perplexity(0.0) == pytest.approx(1.0)
    assert perplexity(1.0) == pytest.approx(math.e)


def test_when_token_accuracy_then_ignores_masked() -> None:
    logits = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
    targets = torch.tensor([[0, -100]])
    assert token_accuracy(logits, targets) == pytest.approx(1.0)


def test_when_nll_computed_then_matches_cross_entropy() -> None:
    torch.manual_seed(1)
    logits = torch.randn(2, 6, 9)
    labels = torch.randint(0, 9, (2, 6))
    labels[0, 2] = -100
    got = negative_log_likelihood(logits, labels)
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 9), labels.reshape(-1), ignore_index=-100
    ).item()
    assert got == pytest.approx(expected)


def test_when_nll_all_masked_then_zero() -> None:
    logits = torch.randn(1, 3, 4)
    labels = torch.full((1, 3), -100)
    assert negative_log_likelihood(logits, labels) == 0.0

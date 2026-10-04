"""Parity tests: LM objective under BPTT and parallel training."""

from __future__ import annotations

from typing import cast

import pytest
import torch

import nonlinearrnnscanbeparallel.models  # noqa: F401
from nonlinearrnnscanbeparallel.models.parallel_wrapper import ParallelRNNTrainer
from nonlinearrnnscanbeparallel.models.registry import get_model
from nonlinearrnnscanbeparallel.models.scaffold import (
    FourierScaffold,
    LinearSumScaffold,
    ScaffoldStack,
)
from nonlinearrnnscanbeparallel.tasks.language_modeling import BPTTLMTask, LMLightningTask


def _nano() -> torch.nn.Module:
    return get_model(
        "nano_rnn",
        input_dim=32,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        vocab_size=64,
        mixer_type="min_gru",
    )


def _batch() -> dict[str, torch.Tensor]:
    ids = torch.randint(0, 64, (2, 12))
    return {"input_ids": ids, "labels": ids.clone()}


def test_when_wrapper_eval_then_matches_target_sequential() -> None:
    target = _nano()
    wrapper = ParallelRNNTrainer(target, chunk_size=4, scaffold_dim=16)
    wrapper.eval()
    ids = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        wrapper_logits = wrapper(ids)
        target_logits, _ = target(ids)
    torch.testing.assert_close(wrapper_logits, target_logits)


def test_when_wrapper_train_forward_then_logits_shape() -> None:
    wrapper = ParallelRNNTrainer(_nano(), chunk_size=4, scaffold_dim=16)
    wrapper.train()
    logits = wrapper(torch.randint(0, 64, (2, 12)))
    assert logits.shape == (2, 12, 64)


def test_when_linear_sum_scaffold_training_step_then_finite_loss() -> None:
    spec = {"vocab_size": 64, "hidden_dim": 32, "bptt_max_seq_len": 12, "total_steps": 10}
    wrapper = ParallelRNNTrainer(
        _nano(), chunk_size=4, scaffold_dim=16, scaffold_type="linear_sum", dropout=0.0
    )
    parallel = LMLightningTask(spec, wrapper)
    parallel.train()
    loss = parallel.training_step(_batch(), 0)
    assert torch.isfinite(loss)
    first_scaffold = cast(ScaffoldStack, wrapper.scaffolds[0])
    assert isinstance(first_scaffold.layers[0], LinearSumScaffold)


def test_when_fourier_scaffold_training_step_then_finite_loss() -> None:
    spec = {"vocab_size": 64, "hidden_dim": 32, "bptt_max_seq_len": 12, "total_steps": 10}
    wrapper = ParallelRNNTrainer(
        _nano(), chunk_size=4, scaffold_dim=16, scaffold_type="fourier", dropout=0.0
    )
    parallel = LMLightningTask(spec, wrapper)
    parallel.train()
    loss = parallel.training_step(_batch(), 0)
    assert torch.isfinite(loss)
    first_scaffold = cast(ScaffoldStack, wrapper.scaffolds[0])
    assert isinstance(first_scaffold.layers[0], FourierScaffold)
    assert first_scaffold.positional is True


def test_when_boundary_states_causal_then_suffix_perturbation_leaves_prefix_boundaries() -> None:
    wrapper = ParallelRNNTrainer(
        _nano(), chunk_size=4, scaffold_dim=16, scaffold_type="fourier", dropout=0.0
    )
    x = torch.randn(2, 12, 32)
    reference = wrapper.get_boundary_states(x)
    perturbed_input = x.clone()
    perturbed_input[:, 8:, :] += 100.0  # only the last chunk's interior
    assert len(reference) == 2  # one per layer
    for layer_output, perturbed_output in zip(
        reference, wrapper.get_boundary_states(perturbed_input), strict=True
    ):
        # Boundary indices for chunk_size=4, T=12 are 3 and 7; neither may
        # move when tokens >= 8 change, since chunk starts seed only from
        # the preceding boundary state.
        torch.testing.assert_close(layer_output, perturbed_output)
    assert layer_output.shape == (2, 2, 32)  # [B, M-1, hidden_dim] translated


def test_when_bptt_and_parallel_training_step_then_finite_loss() -> None:
    spec = {"vocab_size": 64, "hidden_dim": 32, "bptt_max_seq_len": 12, "total_steps": 10}
    bptt = BPTTLMTask(spec, _nano())
    bptt_loss = bptt.training_step(_batch(), 0)
    assert torch.isfinite(bptt_loss)

    wrapper = ParallelRNNTrainer(_nano(), chunk_size=4, scaffold_dim=16)
    parallel = LMLightningTask(spec, wrapper)
    parallel.train()
    parallel_loss = parallel.training_step(_batch(), 0)
    assert torch.isfinite(parallel_loss)


def test_when_lm_tasks_log_then_same_metric_keys() -> None:
    spec = {"vocab_size": 64, "hidden_dim": 32, "bptt_max_seq_len": 12, "total_steps": 10}
    bptt = BPTTLMTask(spec, _nano())
    wrapper = ParallelRNNTrainer(_nano(), chunk_size=4, scaffold_dim=16)
    parallel = LMLightningTask(spec, wrapper)

    keys: dict[str, set[str]] = {}
    for name, task in (("bptt", bptt), ("parallel", parallel)):
        logged: list[str] = []

        def _capture(*args: object, _logged: list[str] = logged, **kwargs: object) -> None:
            _logged.append(str(args[0]) if args else str(kwargs.get("name")))

        task.log = _capture  # type: ignore[method-assign]
        task.validation_step(_batch(), 0)
        keys[name] = set(logged)

    assert keys["bptt"] == keys["parallel"]
    assert keys["bptt"] == {"val/loss", "val/nll", "val/ppl", "val/acc"}


def _hope_nano() -> torch.nn.Module:
    """A HOPE-mixed NanoRNN language model for the parity checks below."""
    return get_model(
        "nano_rnn",
        input_dim=0,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        vocab_size=64,
        mixer_type="hope",
        mem_dim=8,
    )


def test_when_hope_wrapper_evals_then_matches_target_sequential() -> None:
    """Eval bypasses scaffold and translator, so it must be exact.

    HOPE carries its matrix memories plus q and CMS caches; if the wrapper's
    eval path and the target's own forward ever disagreed, every reported
    validation number would be measuring the wrong model.
    """
    torch.manual_seed(0)
    target = _hope_nano()
    wrapper = ParallelRNNTrainer(target, chunk_size=4, scaffold_dim=16)
    wrapper.eval()
    ids = torch.randint(0, 64, (2, 12))

    with torch.no_grad():
        wrapper_logits = wrapper(ids)
        target_logits, _ = target(ids)

    torch.testing.assert_close(wrapper_logits, target_logits)


def test_when_hope_trains_parallel_then_loss_matches_bptt_within_sub_percent() -> None:
    """Both modes start from the same target weights and stay in the same band.

    Exact equality is not expected: the parallel path reconstructs each
    boundary through a randomly initialized scaffold and translator rather
    than running the target recurrence. The repo's parity criterion is a
    sub-percent loss gap; this asserts that HOPE stays inside it, which is
    what makes its validation numbers comparable to a BPTT baseline.
    """
    torch.manual_seed(0)
    spec = {"vocab_size": 64, "hidden_dim": 32, "bptt_max_seq_len": 12, "total_steps": 10}
    target = _hope_nano()
    bptt = BPTTLMTask(spec, target)
    wrapper = ParallelRNNTrainer(target, chunk_size=4, scaffold_dim=16)
    parallel = LMLightningTask(spec, wrapper)
    parallel.train()
    batch = _batch()

    bptt_loss = float(bptt.training_step(batch, 0))
    parallel_loss = float(parallel.training_step(batch, 0))

    assert parallel_loss == pytest.approx(bptt_loss, rel=0.01)


def test_when_hope_bptt_and_parallel_step_then_both_losses_stay_finite() -> None:
    """A real training step runs under both modes without diverging."""
    torch.manual_seed(0)
    spec = {"vocab_size": 64, "hidden_dim": 32, "bptt_max_seq_len": 12, "total_steps": 10}
    bptt = BPTTLMTask(spec, _hope_nano())
    bptt_loss = bptt.training_step(_batch(), 0)
    assert torch.isfinite(bptt_loss)

    wrapper = ParallelRNNTrainer(_hope_nano(), chunk_size=4, scaffold_dim=16)
    parallel = LMLightningTask(spec, wrapper)
    parallel.train()
    parallel_loss = parallel.training_step(_batch(), 0)

    assert torch.isfinite(parallel_loss)


def test_when_hope_validates_then_metrics_match_the_bptt_key_set() -> None:
    """HOPE logs the same LM metrics in both modes, so runs are comparable."""
    torch.manual_seed(0)
    spec = {"vocab_size": 64, "hidden_dim": 32, "bptt_max_seq_len": 12, "total_steps": 10}
    bptt = BPTTLMTask(spec, _hope_nano())
    wrapper = ParallelRNNTrainer(_hope_nano(), chunk_size=4, scaffold_dim=16)
    parallel = LMLightningTask(spec, wrapper)

    keys: dict[str, set[str]] = {}
    for name, task in (("bptt", bptt), ("parallel", parallel)):
        logged: list[str] = []

        def _capture(*args: object, _logged: list[str] = logged, **kwargs: object) -> None:
            _logged.append(str(args[0]) if args else str(kwargs.get("name")))

        task.log = _capture  # type: ignore[method-assign]
        task.validation_step(_batch(), 0)
        keys[name] = set(logged)

    assert keys["bptt"] == keys["parallel"] == {"val/loss", "val/nll", "val/ppl", "val/acc"}

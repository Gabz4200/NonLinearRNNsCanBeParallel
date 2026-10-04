"""Tests for the Trainer factory in :mod:`training.engine`."""

from __future__ import annotations

from typing import Any

import pytest
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

from nonlinearrnnscanbeparallel.training.engine import create_trainer, trainer_callbacks


class _Marker(Callback):
    """A callback that is not in the config-driven registry."""


def test_when_extra_callbacks_given_then_all_appear_on_the_trainer() -> None:
    marker = _Marker()
    trainer = create_trainer(
        {"max_epochs": 1, "callbacks": {"model_checkpoint": True, "early_stopping": True}},
        extra_callbacks=[marker],
    )

    assert marker in trainer_callbacks(trainer)
    assert sum(isinstance(cb, ModelCheckpoint) for cb in trainer_callbacks(trainer)) == 1
    assert sum(isinstance(cb, EarlyStopping) for cb in trainer_callbacks(trainer)) == 1


def test_when_callback_disabled_in_config_then_it_is_not_configured() -> None:
    trainer = create_trainer({"max_epochs": 1, "callbacks": {"model_checkpoint": False}})

    # Lightning always installs its own unmonitored default checkpointing
    # callback, so absence is only observable via the factory's monitor value.
    assert not any(
        isinstance(cb, ModelCheckpoint) and cb.monitor == "val/loss"
        for cb in trainer_callbacks(trainer)
    )


def test_when_extra_callbacks_omitted_then_config_callbacks_still_apply() -> None:
    trainer = create_trainer({"max_epochs": 1, "callbacks": {"model_checkpoint": True}})

    assert (
        sum(
            isinstance(cb, ModelCheckpoint) and cb.monitor == "val/loss"
            for cb in trainer_callbacks(trainer)
        )
        == 1
    )


@pytest.mark.parametrize(
    "extra",
    [pytest.param([None], id="none-entry-skipped")],
)
def test_when_extra_callbacks_contain_none_then_it_is_skipped(extra: list[Any]) -> None:
    trainer = create_trainer({"max_epochs": 1}, extra_callbacks=extra)

    assert not any(cb is None for cb in trainer_callbacks(trainer))

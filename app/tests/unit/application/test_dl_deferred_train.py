import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.deep_learning.dl_deferred_train import (
    cancel_deferred_symbol_training,
    enqueue_deferred_symbol_training,
)


@pytest.mark.asyncio
async def test_deferred_training_chains_next_bootstrap_symbol():
    orch = SimpleNamespace()
    train_fn = MagicMock(return_value=(object(), 0.1))

    with (
        patch(
            "src.application.services.deep_learning.dl_deferred_train.asyncio.to_thread",
            new_callable=AsyncMock,
        ),
        patch(
            "src.application.services.deep_learning.dl_deferred_train.try_enqueue_next_bootstrap_training"
        ) as mock_chain,
    ):
        enqueue_deferred_symbol_training(
            orch,
            "R_10",
            train_fn=train_fn,
            train_args=("R_10",),
            train_kwargs={"granularity": 60},
        )
        await orch._dl_deferred_tasks["R_10"]

    mock_chain.assert_called_once_with(orch)


@pytest.mark.asyncio
async def test_enqueue_runs_training_in_background():
    orch = SimpleNamespace()
    train_fn = MagicMock(return_value=(object(), 0.1))

    with patch(
        "src.application.services.deep_learning.dl_deferred_train.asyncio.to_thread",
        new_callable=AsyncMock,
    ) as mock_thread:
        enqueue_deferred_symbol_training(
            orch,
            "R_10",
            train_fn=train_fn,
            train_args=("R_10",),
            train_kwargs={"granularity": 60},
        )
        task = orch._dl_deferred_tasks["R_10"]
        await task

    mock_thread.assert_awaited_once()
    assert "R_10" not in (getattr(orch, "_dl_force_retrain", None) or {})


@pytest.mark.asyncio
async def test_enqueue_skips_when_task_pending():
    orch = SimpleNamespace()
    pending = MagicMock()
    pending.done.return_value = False
    orch._dl_deferred_tasks = {"R_10": pending}
    train_fn = MagicMock()

    with patch("src.application.services.deep_learning.dl_deferred_train.asyncio.create_task") as mock_create:
        enqueue_deferred_symbol_training(
            orch,
            "R_10",
            train_fn=train_fn,
            train_args=(),
            train_kwargs={},
        )

    mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_enqueue_logs_training_failure():
    orch = SimpleNamespace()
    train_fn = MagicMock()

    with (
        patch(
            "src.application.services.deep_learning.dl_deferred_train.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=RuntimeError("train_fail"),
        ),
        patch("src.application.services.deep_learning.dl_deferred_train.logger") as mock_logger,
    ):
        enqueue_deferred_symbol_training(
            orch,
            "R_10",
            train_fn=train_fn,
            train_args=(),
            train_kwargs={},
        )
        await orch._dl_deferred_tasks["R_10"]

    mock_logger.error.assert_called_once()


def test_cancel_deferred_symbol_training():
    orch = SimpleNamespace()
    pending = MagicMock()
    pending.done.return_value = False
    orch._dl_deferred_tasks = {"R_10": pending}
    cancel_deferred_symbol_training(orch)
    pending.cancel.assert_called_once()
    assert orch._dl_deferred_tasks == {}


@pytest.mark.asyncio
async def test_enqueue_defers_when_another_symbol_training():
    orch = SimpleNamespace()

    def slow_train(*_args, **_kwargs):
        time.sleep(0.2)

    with patch(
        "src.application.services.deep_learning.dl_deferred_train.asyncio.to_thread",
        new_callable=AsyncMock,
        side_effect=slow_train,
    ):
        enqueue_deferred_symbol_training(
            orch,
            "R_10",
            train_fn=MagicMock(),
            train_args=(),
            train_kwargs={},
        )
        enqueue_deferred_symbol_training(
            orch,
            "R_50",
            train_fn=MagicMock(),
            train_args=(),
            train_kwargs={},
        )
        assert "R_50" not in orch._dl_deferred_tasks
        assert "R_10" in orch._dl_deferred_tasks
        await orch._dl_deferred_tasks["R_10"]


@pytest.mark.asyncio
async def test_enqueue_skips_duplicate_pending_symbol():
    orch = SimpleNamespace()
    pending = MagicMock()
    pending.done.return_value = False
    orch._dl_deferred_tasks = {"R_10": pending}

    with patch("src.application.services.deep_learning.dl_deferred_train.asyncio.create_task") as mock_create:
        enqueue_deferred_symbol_training(
            orch,
            "R_10",
            train_fn=MagicMock(),
            train_args=(),
            train_kwargs={},
        )
        mock_create.assert_not_called()


def test_cancel_noop_without_tasks():
    orch = SimpleNamespace()
    cancel_deferred_symbol_training(orch)

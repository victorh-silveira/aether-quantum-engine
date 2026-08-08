from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_bootstrap_train import (
    _history_wait_seconds,
    _ordered_bootstrap_symbols,
    _train_bootstrap_symbol,
    run_dl_training_session,
    run_initial_bootstrap_training,
)
from tests.market_symbols import ALT_SYMBOL, ANCHOR


def test_history_wait_seconds_caps_m15_granularity():
    assert _history_wait_seconds(900) == pytest.approx(30.0)
    assert _history_wait_seconds(15) == pytest.approx(15.0)
    assert _history_wait_seconds(900, {"bootstrap_history_wait_cap_seconds": 10}) == pytest.approx(10.0)
    assert _history_wait_seconds(900, {"bootstrap_history_wait_cap_seconds": "x"}) == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_run_initial_bootstrap_training_sequences_symbols(orch_ready):
    orch = orch_ready
    trained: list[str] = []

    async def fake_train(_orch, symbol: str) -> str:
        trained.append(symbol)
        return "ok"

    with (
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train._ordered_bootstrap_symbols",
            side_effect=lambda _orch: [] if len(trained) >= 2 else [ANCHOR, ALT_SYMBOL],
        ),
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
            side_effect=fake_train,
        ) as mock_train,
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train.runtime_in_training",
            return_value=True,
        ),
    ):
        await run_initial_bootstrap_training(orch)

    assert trained == [ANCHOR, ALT_SYMBOL]
    assert mock_train.await_count == 2


def test_ordered_bootstrap_symbols_empty_when_none_pending(orch_ready):
    orch = orch_ready
    with patch(
        "src.application.services.deep_learning.dl_bootstrap_train.training_priority_symbols",
        return_value=frozenset(),
    ):
        assert _ordered_bootstrap_symbols(orch) == []


def test_ordered_bootstrap_symbols_returns_pending_in_config_order(orch_ready):
    orch = orch_ready
    with patch(
        "src.application.services.deep_learning.dl_bootstrap_train.training_priority_symbols",
        return_value=frozenset({"R_10", "R_50"}),
    ):
        assert _ordered_bootstrap_symbols(orch) == ["R_10"]


@pytest.mark.asyncio
async def test_train_bootstrap_symbol_runs_training_in_thread(orch_ready):
    orch = orch_ready
    n = 3000
    ohlc = tuple(np.linspace(1.0, 2.0, n) for _ in range(4))
    runtime: dict = {}

    async def fake_thread(*_args, **_kwargs):
        runtime["export_ok"] = True

    with (
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train._bootstrap_training_context",
            return_value=(
                {},
                {"lookback": 32},
                100,
                60,
                runtime,
                ohlc[0],
                ohlc[1],
                ohlc[2],
                ohlc[3],
                None,
            ),
        ),
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train.asyncio.to_thread",
            side_effect=fake_thread,
        ) as mock_thread,
    ):
        status = await _train_bootstrap_symbol(orch, "R_10")
    assert status == "ok"
    mock_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_initial_bootstrap_training_skips_trained_runtime(orch_ready):
    orch = orch_ready
    with (
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train._ordered_bootstrap_symbols",
            return_value=["R_10"],
        ),
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train.runtime_in_training",
            return_value=False,
        ),
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
            new_callable=AsyncMock,
        ) as mock_train,
    ):
        await run_initial_bootstrap_training(orch)
    mock_train.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_initial_bootstrap_training_waits_for_history(orch_ready):
    orch = orch_ready
    calls = {"n": 0}

    async def fake_train(_orch, symbol: str) -> str:
        calls["n"] += 1
        return "ok" if calls["n"] >= 2 else "wait"

    with (
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train._ordered_bootstrap_symbols",
            side_effect=[["R_10"], ["R_10"], []],
        ),
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
            side_effect=fake_train,
        ),
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train.runtime_in_training",
            return_value=True,
        ),
        patch.object(orch.stream, "ensure_cluster_history", new_callable=AsyncMock),
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep,
    ):
        await run_initial_bootstrap_training(orch)

    mock_sleep.assert_awaited_once()
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_run_initial_bootstrap_training_stops_after_max_wait_rounds(orch_ready):
    orch = orch_ready
    orch.config.setdefault("deep_learning", {})["bootstrap_max_wait_rounds"] = 2
    with (
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train._ordered_bootstrap_symbols",
            return_value=["R_10"],
        ),
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
            new_callable=AsyncMock,
            return_value="wait",
        ),
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train.runtime_in_training",
            return_value=True,
        ),
        patch.object(orch.stream, "ensure_cluster_history", new_callable=AsyncMock),
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep,
    ):
        await run_initial_bootstrap_training(orch)
    assert mock_sleep.await_count == 1


@pytest.mark.asyncio
async def test_run_initial_bootstrap_training_noop_when_nothing_pending(orch_ready):
    orch = orch_ready
    with (
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train._ordered_bootstrap_symbols",
            return_value=[],
        ),
        patch(
            "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
            new_callable=AsyncMock,
        ) as mock_train,
    ):
        await run_initial_bootstrap_training(orch)
    mock_train.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_dl_training_session_sequences_all_symbols(orch_ready):
    orch = orch_ready
    trained: list[str] = []

    async def fake_train(_orch, symbol):
        trained.append(symbol)
        return "ok"

    with patch(
        "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
        side_effect=fake_train,
    ):
        assert await run_dl_training_session(orch) is True
    assert trained == [str(s) for s in orch.symbols]


@pytest.mark.asyncio
async def test_run_dl_training_session_noop_without_symbols(orch_ready):
    orch = orch_ready
    orch.symbols = []
    orch.config.setdefault("deep_learning", {})["train_symbols"] = []
    with patch(
        "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
        new_callable=AsyncMock,
    ) as mock_train:
        assert await run_dl_training_session(orch) is True
    mock_train.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_dl_training_session_waits_for_history(orch_ready):
    orch = orch_ready
    orch.symbols = ["R_10", "R_50"]
    orch.config.setdefault("deep_learning", {})["train_symbols"] = ["R_10", "R_50"]
    attempts = {"R_10": 0, "R_50": 0}

    async def fake_train(_orch, symbol):
        attempts[symbol] += 1
        return "ok" if attempts[symbol] >= 2 else "wait"

    orch.stream.ensure_cluster_history = AsyncMock()
    with patch(
        "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
        side_effect=fake_train,
    ):
        assert await run_dl_training_session(orch) is True
    assert attempts["R_10"] >= 2
    assert attempts["R_50"] >= 2
    orch.stream.ensure_cluster_history.assert_awaited()


@pytest.mark.asyncio
async def test_run_dl_training_session_stops_after_max_wait_rounds(orch_ready):
    orch = orch_ready
    orch.config.setdefault("deep_learning", {})["bootstrap_max_wait_rounds"] = 1
    with patch(
        "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
        new_callable=AsyncMock,
        return_value="wait",
    ):
        assert await run_dl_training_session(orch) is False


@pytest.mark.asyncio
async def test_run_dl_training_session_skips_completed_symbols(orch_ready):
    orch = orch_ready
    orch.symbols = ["R_10", "R_50"]
    orch.config.setdefault("deep_learning", {})["train_symbols"] = ["R_10", "R_50"]
    calls: list[str] = []
    r25_attempts = {"n": 0}

    async def fake_train(_orch, symbol):
        calls.append(symbol)
        if symbol == "R_10":
            return "ok"
        r25_attempts["n"] += 1
        return "ok" if r25_attempts["n"] >= 2 else "wait"

    with patch(
        "src.application.services.deep_learning.dl_bootstrap_train._train_bootstrap_symbol",
        side_effect=fake_train,
    ):
        assert await run_dl_training_session(orch) is True
    assert calls[0] == "R_10" and calls.count("R_10") == 1 and calls.count("R_50") >= 2

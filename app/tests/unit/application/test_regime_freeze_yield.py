import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.meta_direction_flip import SIGNAL_SUSPENDED
from src.application.services.orchestrator.regime_freeze_yield import (
    _REGIME_FREEZE_DEFAULT_YIELD_SECONDS,
    _entry_freeze_active,
    _yield_freeze_delay,
    await_regime_freeze_yield,
    cluster_collect_aborted,
    cluster_freeze_active,
    decisions_signal_suspended,
    propagate_cluster_signal_suspended,
    regime_freeze_yield_seconds,
)
from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready
from src.application.services.orchestrator.warm_up_buffer_guard import (
    STREAM_WARM_UP_DELAY_SECONDS,
    schedule_stream_warm_up_barrier,
    stream_warm_up_active,
)


FREEZE_YIELD_MODULE = "src.application.services.orchestrator.regime_freeze_yield"
TRADING_CYCLE_MODULE = "src.application.services.orchestrator.trading_cycle_entry"


def test_decisions_signal_suspended_detects_frozen_entry():
    decisions = {
        "RDBULL": {"metrics": {"signal_status": SIGNAL_SUSPENDED}},
        "RDBEAR": {"metrics": {"execute": True}},
    }
    assert decisions_signal_suspended(decisions) is True


def test_decisions_signal_suspended_false_for_empty_or_active():
    assert decisions_signal_suspended({}) is False
    assert decisions_signal_suspended({"RDBULL": {"metrics": {"execute": True}}}) is False


def test_cluster_freeze_active_detects_regime_guard_action():
    decisions = {
        "RDBULL": {"metrics": {"regime_guard_action": "FREEZE: SKIP CYCLE"}},
        "RDBEAR": {"metrics": {"execute": True, "trade_score": 0.80}},
    }
    assert cluster_freeze_active(decisions) is True


def test_propagate_cluster_signal_suspended_marks_all_symbols():
    decisions = {
        "RDBULL": {"metrics": {"signal_status": SIGNAL_SUSPENDED}},
        "RDBEAR": {"metrics": {"execute": True}},
    }
    propagate_cluster_signal_suspended(decisions)
    assert decisions["RDBULL"]["metrics"]["signal_status"] == SIGNAL_SUSPENDED
    assert decisions["RDBEAR"]["metrics"]["signal_status"] == SIGNAL_SUSPENDED


def test_cluster_collect_aborted_propagates_and_returns_true():
    decisions = {
        "RDBULL": {"metrics": {"regime_guard_action": "FREEZE: SKIP CYCLE"}},
        "RDBEAR": {"metrics": {"execute": True}},
    }
    assert cluster_collect_aborted(decisions) is True
    assert decisions["RDBEAR"]["metrics"]["signal_status"] == SIGNAL_SUSPENDED


def test_cluster_collect_aborted_false_when_cluster_active():
    decisions = {"RDBULL": {"metrics": {"execute": True, "trade_score": 0.80}}}
    assert cluster_collect_aborted(decisions) is False


def test_cluster_freeze_active_false_for_invalid_decisions():
    assert cluster_freeze_active({}) is False
    assert cluster_freeze_active(None) is False


def test_entry_freeze_active_handles_invalid_shapes():
    assert _entry_freeze_active(None) is False
    assert _entry_freeze_active({"metrics": "invalid"}) is False
    assert _entry_freeze_active({"metrics": {"regime_guard_action": "FREEZE: SKIP CYCLE"}}) is True


def test_propagate_cluster_signal_suspended_creates_missing_metrics():
    decisions = {"RDBULL": {}, "RDBEAR": "invalid"}
    propagate_cluster_signal_suspended(decisions)
    assert decisions["RDBULL"]["metrics"]["signal_status"] == SIGNAL_SUSPENDED
    propagate_cluster_signal_suspended(None)
    assert decisions_signal_suspended("bad") is False
    assert decisions_signal_suspended({"RDBULL": "bad"}) is False


def test_regime_freeze_yield_seconds_falls_back_when_m1_boundary_elapsed():
    past_epoch = int(time.time()) - 120
    orch = SimpleNamespace(_last_epoch=past_epoch)
    assert regime_freeze_yield_seconds(orch) == pytest.approx(_REGIME_FREEZE_DEFAULT_YIELD_SECONDS)


def test_regime_freeze_yield_seconds_uses_m1_boundary():
    orch = SimpleNamespace(_last_epoch=120)
    with patch(f"{FREEZE_YIELD_MODULE}.time.time", return_value=150.0):
        assert regime_freeze_yield_seconds(orch) == pytest.approx(30.0)


def test_regime_freeze_yield_seconds_falls_back_without_epoch():
    orch = SimpleNamespace(_last_epoch=0)
    assert regime_freeze_yield_seconds(orch) == pytest.approx(_REGIME_FREEZE_DEFAULT_YIELD_SECONDS)


@pytest.mark.asyncio
async def test_await_regime_freeze_yield_sleeps_when_suspended():
    orch = SimpleNamespace(running=True, _last_epoch=0)
    decisions = {"RDBULL": {"metrics": {"signal_status": SIGNAL_SUSPENDED}}}
    recorded: list[float] = []

    async def record_sleep(seconds: float) -> None:
        recorded.append(seconds)

    with patch(f"{FREEZE_YIELD_MODULE}._yield_freeze_delay", side_effect=record_sleep):
        delay = await await_regime_freeze_yield(orch, decisions)
    assert delay == pytest.approx(_REGIME_FREEZE_DEFAULT_YIELD_SECONDS)
    assert recorded == [pytest.approx(_REGIME_FREEZE_DEFAULT_YIELD_SECONDS)]


@pytest.mark.asyncio
async def test_await_regime_freeze_yield_does_not_hold_state_lock(orch_ready):
    orch = orch_ready
    orch.running = True
    orch._last_epoch = 0
    decisions = {"RDBULL": {"metrics": {"signal_status": SIGNAL_SUSPENDED}}}
    lock_during_yield: list[bool] = []

    async def record_yield(seconds: float) -> None:
        lock_during_yield.append(orch.state_mgr._state_lock.locked())

    with patch(f"{FREEZE_YIELD_MODULE}._yield_freeze_delay", side_effect=record_yield):
        await await_regime_freeze_yield(orch, decisions)
    assert lock_during_yield == [False]


@pytest.mark.asyncio
async def test_yield_freeze_delay_skips_non_positive_delay():
    with patch(f"{FREEZE_YIELD_MODULE}.asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
        await _yield_freeze_delay(0.0)
    sleep_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_yield_freeze_delay_awaits_positive_delay():
    with patch(f"{FREEZE_YIELD_MODULE}.asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
        await _yield_freeze_delay(2.5)
    sleep_mock.assert_awaited_once_with(2.5)


@pytest.mark.asyncio
async def test_await_regime_freeze_yield_skips_when_not_running():
    orch = SimpleNamespace(running=False, _last_epoch=0)
    decisions = {"RDBULL": {"metrics": {"signal_status": SIGNAL_SUSPENDED}}}
    with patch(f"{FREEZE_YIELD_MODULE}._yield_freeze_delay", new_callable=AsyncMock) as sleep_mock:
        delay = await await_regime_freeze_yield(orch, decisions)
    assert delay == 0.0
    sleep_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_await_regime_freeze_yield_skips_without_suspension():
    orch = SimpleNamespace(running=True, _last_epoch=0)
    decisions = {"RDBULL": {"metrics": {"execute": True}}}
    with patch(f"{FREEZE_YIELD_MODULE}._yield_freeze_delay", new_callable=AsyncMock) as sleep_mock:
        delay = await await_regime_freeze_yield(orch, decisions)
    assert delay == 0.0
    sleep_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_reconnect_warm_up_suspends_cycles_before_regime_freeze(orch_ready):
    orch = orch_ready
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    loop = asyncio.get_running_loop()
    base = loop.time()
    schedule_stream_warm_up_barrier(orch)
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={"RDBULL": {"metrics": {"signal_status": SIGNAL_SUSPENDED}}},
        ) as collect_mock,
        patch(f"{FREEZE_YIELD_MODULE}._yield_freeze_delay", new_callable=AsyncMock) as freeze_sleep,
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch.object(loop, "time", return_value=base + 20.0),
    ):
        ran = await run_trading_cycle_if_ready(orch)
    assert ran is True
    collect_mock.assert_not_awaited()
    freeze_sleep.assert_not_awaited()
    assert stream_warm_up_active(orch, now=base + 20.0) is True


@pytest.mark.asyncio
async def test_post_reconnect_warm_up_releases_cycles_after_delay(orch_ready):
    orch = orch_ready
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    loop = asyncio.get_running_loop()
    base = loop.time()
    orch._stream_warmed_up_at = base + STREAM_WARM_UP_DELAY_SECONDS
    orch.executor.execute_cluster = AsyncMock()
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={"RDBULL": {"metrics": {"calibrated_prob": 0.70, "predicted_payoff_edge": 0.08}}},
        ) as collect_mock,
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch.object(loop, "time", return_value=base + STREAM_WARM_UP_DELAY_SECONDS + 1.0),
    ):
        ran = await run_trading_cycle_if_ready(orch)
    assert ran is True
    collect_mock.assert_awaited_once()
    orch.executor.execute_cluster.assert_awaited_once()
    assert stream_warm_up_active(orch, now=base + STREAM_WARM_UP_DELAY_SECONDS + 1.0) is False

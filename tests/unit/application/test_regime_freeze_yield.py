import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.infra_timing_config import resolve_orchestrator_timing_config
from src.application.services.orchestrator.regime_freeze_yield import (
    _entry_freeze_active,
    await_regime_freeze_yield,
    cluster_collect_aborted,
    cluster_freeze_active,
    decisions_signal_suspended,
    propagate_cluster_signal_suspended,
    regime_freeze_yield_seconds,
)
from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready
from src.application.services.orchestrator.warm_up_buffer_guard import (
    schedule_stream_warm_up_barrier,
    stream_warm_up_active,
)
from src.application.services.regime_micro_freeze import SIGNAL_SUSPENDED


TRADING_CYCLE_MODULE = "src.application.services.orchestrator.trading_cycle_entry"


STREAM_WARM_UP_DELAY_SECONDS = float(resolve_orchestrator_timing_config()["stream_warm_up_delay_seconds"]) or 45.0


def test_decisions_signal_suspended_detects_frozen_entry():
    decisions = {
        "R_10": {"metrics": {"signal_status": SIGNAL_SUSPENDED}},
        "R_50": {"metrics": {"execute": True}},
    }
    assert decisions_signal_suspended(decisions) is True


def test_decisions_signal_suspended_false_for_empty_or_active():
    assert decisions_signal_suspended({}) is False
    assert decisions_signal_suspended({"R_10": {"metrics": {"execute": True}}}) is False


def test_cluster_freeze_active_detects_regime_guard_action():
    decisions = {
        "R_10": {"metrics": {"regime_guard_action": "FREEZE: SKIP CYCLE"}},
        "R_50": {"metrics": {"execute": True, "trade_score": 0.80}},
    }
    assert cluster_freeze_active(decisions) is True


def test_propagate_cluster_signal_suspended_marks_all_symbols():
    decisions = {
        "R_10": {"metrics": {"signal_status": SIGNAL_SUSPENDED}},
        "R_50": {"metrics": {"execute": True}},
    }
    propagate_cluster_signal_suspended(decisions)
    assert decisions["R_10"]["metrics"]["signal_status"] == SIGNAL_SUSPENDED
    assert decisions["R_50"]["metrics"]["signal_status"] == SIGNAL_SUSPENDED


def test_cluster_collect_aborted_never_aborts():
    decisions = {
        "R_10": {"metrics": {"regime_guard_action": "FREEZE: SKIP CYCLE"}},
        "R_50": {"metrics": {"execute": True}},
    }
    assert cluster_collect_aborted(decisions) is False


def test_regime_freeze_yield_seconds_always_zero():
    orch = SimpleNamespace(config={"orchestrator": {"cycle_interval_seconds": 60}}, _last_epoch=120)
    assert regime_freeze_yield_seconds(orch) == 0.0


@pytest.mark.asyncio
async def test_await_regime_freeze_yield_is_noop():
    orch = SimpleNamespace(running=True, config={"orchestrator": {"cycle_interval_seconds": 60}}, _last_epoch=0)
    decisions = {"R_10": {"metrics": {"signal_status": SIGNAL_SUSPENDED}}}
    delay = await await_regime_freeze_yield(orch, decisions)
    assert delay == 0.0


@pytest.mark.asyncio
async def test_await_regime_freeze_yield_does_not_hold_state_lock(orch_ready):
    orch = orch_ready
    orch.running = True
    orch._last_epoch = 0
    decisions = {"R_10": {"metrics": {"signal_status": SIGNAL_SUSPENDED}}}
    delay = await await_regime_freeze_yield(orch, decisions)
    assert delay == 0.0
    assert orch.state_mgr._state_lock.locked() is False


@pytest.mark.asyncio
async def test_await_regime_freeze_yield_skips_when_not_running():
    orch = SimpleNamespace(running=False, _last_epoch=0)
    decisions = {"R_10": {"metrics": {"signal_status": SIGNAL_SUSPENDED}}}
    delay = await await_regime_freeze_yield(orch, decisions)
    assert delay == 0.0


@pytest.mark.asyncio
async def test_await_regime_freeze_yield_skips_without_suspension():
    orch = SimpleNamespace(running=True, _last_epoch=0)
    decisions = {"R_10": {"metrics": {"execute": True}}}
    delay = await await_regime_freeze_yield(orch, decisions)
    assert delay == 0.0


def test_entry_freeze_active_branches():
    assert _entry_freeze_active("bad") is False
    assert _entry_freeze_active({"metrics": {"signal_status": SIGNAL_SUSPENDED}}) is True
    assert _entry_freeze_active({"metrics": "bad"}) is False


def test_regime_freeze_helpers_handle_invalid_shapes():
    propagate_cluster_signal_suspended(None)
    propagate_cluster_signal_suspended({"R_10": "bad"})
    assert decisions_signal_suspended("bad") is False
    assert decisions_signal_suspended({"R_10": "bad"}) is False
    assert cluster_freeze_active(None) is False


@pytest.mark.asyncio
async def test_post_reconnect_warm_up_suspends_cycles_before_regime_freeze(orch_ready):
    orch = orch_ready
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    orch.config.setdefault("orchestrator", {})["stream_warm_up_delay_seconds"] = 45.0
    loop = asyncio.get_running_loop()
    base = loop.time()
    schedule_stream_warm_up_barrier(orch)
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={"R_10": {"metrics": {"signal_status": SIGNAL_SUSPENDED}}},
        ) as collect_mock,
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch.object(loop, "time", return_value=base + 20.0),
    ):
        ran = await run_trading_cycle_if_ready(orch)
    assert ran is True
    collect_mock.assert_not_awaited()
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
            return_value={"R_10": {"metrics": {"calibrated_prob": 0.70, "predicted_payoff_edge": 0.08}}},
        ) as collect_mock,
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch.object(loop, "time", return_value=base + STREAM_WARM_UP_DELAY_SECONDS + 1.0),
    ):
        ran = await run_trading_cycle_if_ready(orch)
    assert ran is True
    collect_mock.assert_awaited_once()
    orch.executor.execute_cluster.assert_awaited_once()
    assert stream_warm_up_active(orch, now=base + STREAM_WARM_UP_DELAY_SECONDS + 1.0) is False

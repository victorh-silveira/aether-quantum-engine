import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.meta_direction_flip import SIGNAL_SUSPENDED
from src.application.services.orchestrator.regime_freeze_yield import (
    _REGIME_FREEZE_DEFAULT_YIELD_SECONDS,
    await_regime_freeze_yield,
    decisions_signal_suspended,
    regime_freeze_yield_seconds,
)


FREEZE_YIELD_MODULE = "src.application.services.orchestrator.regime_freeze_yield"


def test_decisions_signal_suspended_detects_frozen_entry():
    decisions = {
        "RDBULL": {"metrics": {"signal_status": SIGNAL_SUSPENDED}},
        "RDBEAR": {"metrics": {"execute": True}},
    }
    assert decisions_signal_suspended(decisions) is True


def test_decisions_signal_suspended_false_for_empty_or_active():
    assert decisions_signal_suspended({}) is False
    assert decisions_signal_suspended({"RDBULL": {"metrics": {"execute": True}}}) is False
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

    with patch(f"{FREEZE_YIELD_MODULE}.asyncio.sleep", side_effect=record_sleep):
        delay = await await_regime_freeze_yield(orch, decisions)
    assert delay == pytest.approx(_REGIME_FREEZE_DEFAULT_YIELD_SECONDS)
    assert recorded == [pytest.approx(_REGIME_FREEZE_DEFAULT_YIELD_SECONDS)]


@pytest.mark.asyncio
async def test_await_regime_freeze_yield_skips_when_not_running():
    orch = SimpleNamespace(running=False, _last_epoch=0)
    decisions = {"RDBULL": {"metrics": {"signal_status": SIGNAL_SUSPENDED}}}
    with patch(f"{FREEZE_YIELD_MODULE}.asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
        delay = await await_regime_freeze_yield(orch, decisions)
    assert delay == 0.0
    sleep_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_await_regime_freeze_yield_skips_without_suspension():
    orch = SimpleNamespace(running=True, _last_epoch=0)
    decisions = {"RDBULL": {"metrics": {"execute": True}}}
    with patch(f"{FREEZE_YIELD_MODULE}.asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
        delay = await await_regime_freeze_yield(orch, decisions)
    assert delay == 0.0
    sleep_mock.assert_not_awaited()

"""Testes de recuperacao transparente e timeouts resilientes pos-liquidacao."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.orchestrator_run_loop import run_orchestrator_main_loop
from src.application.services.orchestrator.post_settlement_cycle import _attempt_post_settlement_trading_cycle
from src.application.services.orchestrator.post_settlement_resilience import (
    clear_post_settlement_polling_state,
    recover_post_settlement_loop_transparently,
    resolve_post_settlement_cycle_timeout,
)
from src.infrastructure.state.trading_state import TradingState


@pytest.fixture
def orchestrator_config(orch_config):
    return orch_config


def test_resolve_post_settlement_cycle_timeout_returns_none_for_meta_zscore_reject(orch_ready):
    orch = orch_ready
    orch._last_quality_gate_regime = "meta_zscore_reject"
    assert resolve_post_settlement_cycle_timeout(orch, {}) is None


def test_resolve_post_settlement_cycle_timeout_returns_none_for_mandatory_continuous(orch_ready):
    orch = orch_ready
    orch._last_quality_gate_regime = "mandatory_continuous"
    assert resolve_post_settlement_cycle_timeout(orch, {}) is None


def test_resolve_post_settlement_cycle_timeout_returns_none_for_mandatory_trade_flag(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = True
    assert resolve_post_settlement_cycle_timeout(orch, {}) is None


def test_resolve_post_settlement_cycle_timeout_uses_configured_limit(orch_ready):
    orch = orch_ready
    orch_cfg = {"post_settlement_cycle_timeout_seconds": 420}
    assert resolve_post_settlement_cycle_timeout(orch, orch_cfg) == 420.0


def test_clear_post_settlement_polling_state_cancels_poll_task(orch_ready):
    orch = orch_ready
    orch.is_trading = True
    poll_task = MagicMock()
    poll_task.done.return_value = False
    orch._trading_slot_poll_task = poll_task
    orch._post_settlement_wake = asyncio.Event()
    orch._post_settlement_wake.set()
    clear_post_settlement_polling_state(orch)
    assert orch.is_trading is False
    assert orch._trading_slot_poll_task is None
    assert orch._post_settlement_wake.is_set() is False
    poll_task.cancel.assert_called_once()


def test_recover_post_settlement_loop_transparently_resets_deadlock(orch_ready, caplog):
    orch = orch_ready
    orch._post_settlement_deadlock = True
    orch._post_settlement_incomplete_streak = 2
    orch.is_trading = True
    with caplog.at_level("INFO"):
        recover_post_settlement_loop_transparently(orch)
    assert orch._post_settlement_deadlock is False
    assert orch._post_settlement_incomplete_streak == 0
    assert orch.is_trading is False
    assert "Loop reinicializado de forma transparente" in caplog.text


def test_recover_post_settlement_loop_transparently_noop_below_limit(orch_ready):
    orch = orch_ready
    orch._post_settlement_incomplete_streak = 1
    recover_post_settlement_loop_transparently(orch)
    assert orch._post_settlement_incomplete_streak == 1


@pytest.mark.asyncio
async def test_attempt_post_settlement_trading_cycle_without_timeout_when_patient(orch_ready):
    orch = orch_ready
    orch._last_quality_gate_regime = "meta_zscore_reject"
    orch_cfg = {"post_settlement_cycle_timeout_seconds": 0.01}
    cycle_mock = AsyncMock(return_value=True)
    with patch.object(orch, "_run_trading_cycle_if_ready", cycle_mock):
        result = await _attempt_post_settlement_trading_cycle(orch, orch_cfg)
    cycle_mock.assert_awaited_once()
    assert result is True


@pytest.mark.asyncio
async def test_orchestrator_run_loop_recovers_instead_of_sys_exit(orchestrator_config, caplog):
    TradingState.reset()
    iterations = 0

    async def stop_after_recovery(*_args, **_kwargs):
        nonlocal iterations
        iterations += 1
        if iterations >= 2:
            orch.running = False

    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch.ws.is_running = True
        orch.running = True
        orch._post_settlement_deadlock = True
        orch._post_settlement_incomplete_streak = 2
        with (
            patch(
                "src.application.services.orchestrator.orchestrator_run_loop.setup_session",
                AsyncMock(return_value=True),
            ),
            patch(
                "src.application.services.orchestrator.orchestrator_run_loop.start_streams",
                AsyncMock(return_value=True),
            ),
            patch(
                "src.application.services.orchestrator.orchestrator_run_loop.start_settlement_worker",
                AsyncMock(),
            ),
            patch(
                "src.application.services.orchestrator.orchestrator_run_loop.start_ingestion_watchdog",
                AsyncMock(),
            ),
            patch(
                "src.application.services.orchestrator.orchestrator_run_loop.prepare_orchestrator_run_loop",
            ),
            patch.object(orch, "_run_trading_cycle_if_ready", AsyncMock(return_value=False)),
            patch.object(orch, "_tick_idle_cycle_watchdog", side_effect=stop_after_recovery),
            patch.object(orch, "_tick_interval_cycle_if_due", AsyncMock()),
            patch(
                "src.application.services.orchestrator.orchestrator_run_loop.get_data_state_signature",
                return_value="sig",
            ),
            caplog.at_level("INFO"),
        ):
            orch.last_data_signature = "sig"
            await run_orchestrator_main_loop(orch)
        assert orch._post_settlement_deadlock is False
        assert orch._post_settlement_incomplete_streak == 0
        assert "Loop reinicializado de forma transparente" in caplog.text

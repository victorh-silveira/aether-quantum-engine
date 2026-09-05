"""Testes de encerramento forcado e curto-circuito stop-win no orquestrador."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.orchestrator_run_loop import (
    _enforce_post_settlement_deadlock_exit,
    emergency_save_session_state,
)
from src.application.services.orchestrator.post_settlement_cycle import run_post_settlement_breath_and_cycle
from src.infrastructure.state.trading_state import TradingState


@pytest.fixture
def orchestrator_config(orch_config):
    return orch_config


def test_emergency_save_session_state_persists_risk_bundle(orchestrator_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch.state.balance = 1105.0
        orch.risk_manager.total_session_profit = 105.0
        orch.state_mgr.reset_session_metrics(1000.0, 101.83)
        orch.state_mgr.persistence.save = MagicMock()
        emergency_save_session_state(orch)
        payload = orch.state_mgr.persistence.save.call_args[0][0]
        assert payload["emergency_shutdown"] is True
        assert payload["total_session_profit"] == 105.0
        assert "risk" in payload


def test_enforce_post_settlement_schedules_passive_reconcile(orchestrator_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch._post_settlement_deadlock = True
        orch._post_settlement_incomplete_streak = 2
        orch.logger = MagicMock()
        with patch("src.application.services.orchestrator.orchestrator_run_loop.spawn_background") as spawn_bg:
            _enforce_post_settlement_deadlock_exit(orch)
        spawn_bg.assert_called_once()
        assert orch._post_settlement_deadlock is False
        assert orch._post_settlement_incomplete_streak == 0
        orch.logger.info.assert_any_call(
            "SETTLE.settle_reconcile: incompleto pos-liquidacao (streak=%d); reconciliacao passiva via portfolio",
            2,
        )


def test_enforce_post_settlement_without_running_loop(orchestrator_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch._post_settlement_deadlock = True
        orch._post_settlement_incomplete_streak = 2
        orch.logger = MagicMock()
        with patch(
            "src.application.services.orchestrator.orchestrator_run_loop.spawn_background",
            side_effect=RuntimeError("no loop"),
        ):
            _enforce_post_settlement_deadlock_exit(orch)
        assert orch._post_settlement_deadlock is False


def test_emergency_save_session_state_sets_initial_from_risk(orchestrator_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch.state.balance = 80.0
        orch.risk_manager.initial_bankroll = 75.0
        orch.risk_manager.total_session_profit = -5.0
        orch.state_mgr.state.initial_balance = 0.0
        orch.state_mgr.persistence.save = MagicMock()
        emergency_save_session_state(orch)
        payload = orch.state_mgr.persistence.save.call_args[0][0]
        assert payload["initial_balance"] == pytest.approx(75.0)


@pytest.mark.asyncio
async def test_orchestrator_run_loop_recovers_on_post_settlement_deadlock(orchestrator_config, caplog):
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
        orch.state_mgr.persistence.save = MagicMock()
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
            patch(
                "src.application.services.orchestrator.orchestrator_run_loop.await_stream_warm_up_gate",
                AsyncMock(return_value=True),
            ),
            patch.object(orch, "_run_trading_cycle_if_ready", AsyncMock(return_value=False)),
            patch.object(orch, "_tick_idle_cycle_watchdog", side_effect=stop_after_recovery),
            patch.object(orch, "_tick_interval_cycle_if_due", AsyncMock()),
            patch(
                "src.application.services.orchestrator.orchestrator_run_loop.get_data_state_signature",
                return_value="sig",
            ),
            patch(
                "src.application.services.orchestrator.orchestrator_run_loop._run_passive_settlement_reconcile",
                AsyncMock(return_value=None),
            ),
            caplog.at_level("INFO"),
        ):
            orch.last_data_signature = "sig"
            await orch.run()
        assert orch._post_settlement_deadlock is False
        assert orch._post_settlement_incomplete_streak == 0
        assert "Loop reinicializado de forma transparente" in caplog.text
        assert "reconciliacao passiva via portfolio" in caplog.text


@pytest.mark.asyncio
async def test_orchestrator_stop_win_short_circuit_with_stuck_settlement_queue(orchestrator_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch.running = True
        orch.state.balance = 1060.0
        orch.state_mgr.reset_session_metrics(1000.0, 50.0)
        orch.state_mgr.state.total_trades_today = 2
        orch.risk_manager.total_session_profit = 60.0
        orch._settlement_queue = asyncio.Queue()
        orch._settlement_queue.put_nowait({"proposal_open_contract": {"contract_id": 99}})
        stuck_worker = asyncio.get_event_loop().create_future()
        orch._settlement_worker_task = stuck_worker
        with patch(
            "src.application.services.orchestrator.post_settlement_cycle.graceful_shutdown",
            new_callable=AsyncMock,
        ) as shutdown_mock:
            await run_post_settlement_breath_and_cycle(orch)
        shutdown_mock.assert_awaited_once()
        assert shutdown_mock.await_args.kwargs["fast_path"] is True
        assert orch.shutdown_reason == "stop_win"


def test_emergency_save_session_state_skips_without_state_manager(orchestrator_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch.state_mgr = None
        emergency_save_session_state(orch)

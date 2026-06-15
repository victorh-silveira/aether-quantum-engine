from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.training_run import run_orchestrator_training
from src.infrastructure.state.trading_state import TradingState


@pytest.mark.asyncio
async def test_orchestrator_run_schedules_bootstrap_in_execute_mode(orch_config):
    TradingState.reset()

    async def stop_loop_after_first_sleep(_seconds):
        orch.running = False

    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        mock_ws_class.return_value.is_running = True
        orch = Orchestrator(orch_config, "token")
        orch._setup_session = AsyncMock(return_value=True)
        orch._start_streams = AsyncMock(return_value=True)
        orch._run_trading_cycle_if_ready = AsyncMock(return_value=True)
        with (
            patch(
                "src.application.services.orchestrator.run_orchestrator_training",
                new_callable=AsyncMock,
            ) as mock_train,
            patch(
                "src.application.services.orchestrator.trading_cycle_entry.try_enqueue_next_bootstrap_training",
            ) as mock_enqueue,
            patch("asyncio.sleep", side_effect=stop_loop_after_first_sleep),
        ):
            await orch.run()
        mock_train.assert_not_called()
        mock_enqueue.assert_called_once()
        assert orch._dl_bootstrap_completed is False


@pytest.mark.asyncio
async def test_orchestrator_run_training_executes_session(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        with patch(
            "src.application.services.orchestrator.run_orchestrator_training",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_train:
            ok = await orch.run_training()
        assert ok is True
        mock_train.assert_awaited_once_with(orch)


@pytest.mark.asyncio
async def test_run_orchestrator_training_success(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._setup_session = AsyncMock(return_value=True)
        orch._start_streams = AsyncMock(return_value=True)
        orch._save_full_state = AsyncMock()
        orch.stop = AsyncMock()
        with patch(
            "src.application.services.orchestrator.training_run.run_dl_training_session",
            new_callable=AsyncMock,
        ) as mock_train:
            ok = await run_orchestrator_training(orch)
        assert ok is True
        mock_train.assert_awaited_once()
        assert orch._dl_bootstrap_completed is True
        orch.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_orchestrator_training_fails_when_setup_fails(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._setup_session = AsyncMock(return_value=False)
        ok = await run_orchestrator_training(orch)
        assert ok is False


@pytest.mark.asyncio
async def test_run_orchestrator_training_fails_when_streams_fail(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._setup_session = AsyncMock(return_value=True)
        orch._start_streams = AsyncMock(return_value=False)
        ok = await run_orchestrator_training(orch)
        assert ok is False


@pytest.mark.asyncio
async def test_orchestrator_on_candle_ticks_retrain_counters_in_train_mode(orch_config, orch_config_train):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config_train, "token")
        orch.anchor = "R_50"
        candle = MagicMock()
        candle.symbol = "R_50"
        candle.epoch = 100
        orch._last_epoch = 0
        with patch("src.application.services.orchestrator.tick_bars_since_train") as mock_tick:
            await orch._on_candle(candle)
        mock_tick.assert_called_once_with(orch, orch.symbols)

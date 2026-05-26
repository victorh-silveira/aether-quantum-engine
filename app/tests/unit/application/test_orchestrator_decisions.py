import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator


@pytest.mark.asyncio
async def test_on_candle_throttling_and_cooldown(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        candle = MagicMock(symbol="frxEURUSD")

        orch.risk_manager.is_on_cooldown = MagicMock(return_value=False)
        await orch._on_candle(candle)
        assert orch.tick_count == 1

        orch.risk_manager.is_on_cooldown = MagicMock(return_value=True)
        await orch._on_candle(candle)
        assert orch.tick_count == 2


@pytest.mark.asyncio
async def test_run_trading_cycle_requires_llm_enabled(orch_config):
    orch_config["llm"] = {"enabled": False}
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        orch.stream.is_synchronized = True
        orch.ws.is_running = True
        orch.logger = MagicMock()
        orch.executor.execute_cluster = AsyncMock()
        await orch._run_trading_cycle_if_ready()
        orch.executor.execute_cluster.assert_not_called()
        orch.logger.error.assert_called()


@pytest.mark.asyncio
async def test_orchestrator_on_candle_extra_coverage(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")

        candle = MagicMock(symbol="OTHER_SYM", epoch=1000)
        await orch._on_candle(candle)
        assert orch.tick_count == 0

        candle.symbol = orch.anchor
        orch.stream.is_synchronized = True
        orch.ws.is_running = True
        orch.executor.execute_cluster = AsyncMock()
        await orch._on_candle(candle)
        orch.executor.execute_cluster.assert_called_once()

        orch.executor.execute_cluster.side_effect = Exception("FAIL")
        await orch._on_candle(MagicMock(symbol=orch.anchor, epoch=1001))


@pytest.mark.asyncio
async def test_on_candle_returns_when_is_trading(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        orch.is_trading = True
        orch.risk_manager.is_on_cooldown = MagicMock(return_value=False)
        orch._last_epoch = 0
        orch.executor.execute_cluster = AsyncMock()
        await orch._on_candle(MagicMock(symbol=orch.anchor, epoch=8888))
        orch.executor.execute_cluster.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_symbols_list_preserves_order_when_anchor_included(orch_config):
    orch_config["symbols"] = ["frxEURUSD", "frxEURUSD", "OTC_SPC"]
    orch_config["anchor"] = "frxEURUSD"
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        assert orch.symbols == ["frxEURUSD", "OTC_SPC"]


@pytest.mark.asyncio
async def test_orchestrator_symbols_list_prepends_anchor_when_missing(orch_config):
    orch_config["symbols"] = ["frxEURUSD", "OTC_SPC"]
    orch_config["anchor"] = "OTC_FCHI"
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        assert orch.symbols[0] == "OTC_FCHI"
        assert set(orch.symbols) == {"OTC_FCHI", "frxEURUSD", "OTC_SPC"}


@pytest.mark.asyncio
async def test_orchestrator_symbols_default_from_single_fallback(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch_config.pop("symbols", None)
        orch_config["anchor"] = "frxEURUSD"
        orch = Orchestrator(orch_config, "token")
        assert orch.symbols == ["frxEURUSD"]


@pytest.mark.asyncio
async def test_interval_gate_calls_run_trading_cycle_when_due(orch_config):
    orch_config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 30
    orch_config.setdefault("llm", {})["refresh_schedule"] = "always"
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        orch.stream.is_synchronized = True
        orch.ws.is_running = True
        orch._last_cluster_cycle_end = 0.0
        orch._run_trading_cycle_if_ready = AsyncMock()
        with patch("src.application.services.orchestrator.time.time", return_value=100.0):
            await orch._tick_interval_cycle_if_due()
        orch._run_trading_cycle_if_ready.assert_called_once()


def test_schedule_trading_cycle_early_returns(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        orch.running = False
        orch.schedule_trading_cycle_after_settlement()

        orch.running = True
        orch.state.active_contracts = {1: object()}
        orch.schedule_trading_cycle_after_settlement()

        orch.state.active_contracts = {}
        orch.is_trading = True
        orch.schedule_trading_cycle_after_settlement()

        orch.is_trading = False
        pending = MagicMock()
        pending.done.return_value = False
        orch._post_settlement_task = pending
        with patch("asyncio.create_task") as mock_create:
            orch.schedule_trading_cycle_after_settlement()
            mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_schedule_trading_cycle_after_settlement_spawns_task(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        orch.running = True
        orch.state.active_contracts = {}
        orch._run_trading_cycle_if_ready = AsyncMock()
        orch.schedule_trading_cycle_after_settlement()
        await asyncio.sleep(0.05)
        orch._run_trading_cycle_if_ready.assert_called_once()


@pytest.mark.asyncio
async def test_interval_gate_runs_with_tag_change_schedule(orch_config):
    orch_config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 30
    orch_config.setdefault("llm", {})["refresh_schedule"] = "tag_change"
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        orch.stream.is_synchronized = True
        orch.ws.is_running = True
        orch._last_cluster_cycle_end = 0.0
        orch._run_trading_cycle_if_ready = AsyncMock()
        with patch("src.application.services.orchestrator.time.time", return_value=100.0):
            await orch._tick_interval_cycle_if_due()
        orch._run_trading_cycle_if_ready.assert_called_once()


def test_mark_cluster_cycle_complete_sets_timestamp(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        with patch("src.application.services.orchestrator.time.time", return_value=42.5):
            orch.mark_cluster_cycle_complete()
        assert orch._last_cluster_cycle_end == 42.5

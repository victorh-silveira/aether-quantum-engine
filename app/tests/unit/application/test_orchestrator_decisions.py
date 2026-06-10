from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator


@pytest.mark.asyncio
async def test_run_trading_cycle_waits_for_open_contract_settlement(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        orch.stream.is_synchronized = True
        orch.ws.is_running = True
        orch.state.active_contracts = {99: object()}
        orch.logger = MagicMock()
        orch.executor.execute_cluster = AsyncMock()
        await orch._run_trading_cycle_if_ready()
        orch.executor.execute_cluster.assert_not_called()
        assert orch._settlement_wait_logged is True


@pytest.mark.asyncio
async def test_on_candle_throttling_and_cooldown(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        candle = MagicMock(symbol="R_50")

        orch.risk_manager.is_on_cooldown = MagicMock(return_value=False)
        await orch._on_candle(candle)
        assert orch.tick_count == 1

        orch.risk_manager.is_on_cooldown = MagicMock(return_value=True)
        await orch._on_candle(candle)
        assert orch.tick_count == 2


@pytest.mark.asyncio
async def test_run_trading_cycle_requires_dl_enabled(orch_config):
    orch_config["deep_learning"] = {"enabled": False}
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
    orch_config.pop("strategy", None)
    orch_config["symbols"] = ["R_50", "R_50", "R_75"]
    orch_config["anchor"] = "R_50"
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        assert orch.symbols == ["R_50", "R_75"]


@pytest.mark.asyncio
async def test_orchestrator_symbols_list_prepends_anchor_when_missing(orch_config):
    orch_config.pop("strategy", None)
    orch_config["symbols"] = ["R_50", "R_75"]
    orch_config["anchor"] = "R_50"
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        assert orch.symbols[0] == "R_50"
        assert set(orch.symbols) == {"R_50", "R_75"}


@pytest.mark.asyncio
async def test_orchestrator_symbols_default_from_single_fallback(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch_config.pop("strategy", None)
        orch_config.pop("symbols", None)
        orch_config["anchor"] = "R_50"
        orch = Orchestrator(orch_config, "token")
        assert orch.symbols == ["R_50"]


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


@pytest.mark.asyncio
async def test_interval_gate_skips_when_post_settlement_pending(orch_config):
    orch_config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 30
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        orch.stream.is_synchronized = True
        orch._last_cluster_cycle_end = 0.0
        pending = MagicMock()
        pending.done.return_value = False
        orch._post_settlement_task = pending
        orch._run_trading_cycle_if_ready = AsyncMock()
        with patch("src.application.services.orchestrator.time.time", return_value=100.0):
            await orch._tick_interval_cycle_if_due()
        orch._run_trading_cycle_if_ready.assert_not_called()


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


@pytest.mark.asyncio
async def test_run_trading_cycle_inserts_blank_line_between_cycles(orch_config):
    with (
        patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()),
        patch(
            "src.application.services.orchestrator.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={},
        ),
    ):
        orch = Orchestrator(orch_config, "token")
        orch.stream.is_synchronized = True
        orch.ws.is_running = True
        orch.logger = MagicMock()
        orch.executor.execute_cluster = AsyncMock()
        await orch._run_trading_cycle_if_ready()
        assert not any(c.args == ("",) for c in orch.logger.info.call_args_list)
        await orch._run_trading_cycle_if_ready()
        blank_calls = [c for c in orch.logger.info.call_args_list if c.args == ("",)]
        assert len(blank_calls) == 1
        await orch._run_trading_cycle_if_ready()
        blank_calls = [c for c in orch.logger.info.call_args_list if c.args == ("",)]
        assert len(blank_calls) == 2


@pytest.mark.asyncio
async def test_run_trading_cycle_lock_recheck_is_trading(orch_config):
    with (
        patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()),
        patch(
            "src.application.services.orchestrator.acquire_trading_cycle_lock",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        orch = Orchestrator(orch_config, "token")
        orch.stream.is_synchronized = True
        orch.ws.is_running = True
        orch.executor.execute_cluster = AsyncMock()
        result = await orch._run_trading_cycle_if_ready()
        assert result is False
        orch.executor.execute_cluster.assert_not_awaited()

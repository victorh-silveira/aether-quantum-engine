"""Testes unitarios para o ciclo de vida do Orquestrador (motor podado)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.infrastructure.api.deriv_rest_client import DerivRestError, DerivTradingSession
from src.infrastructure.state.trading_state import TradingState


@pytest.fixture
def orchestrator_config(orch_config):
    """Reusa a fixture orch_config para o ciclo de vida."""
    return orch_config


@pytest.mark.asyncio
async def test_orchestrator_setup_and_auth(orchestrator_config):
    """Testa a configuração do orquestrador e autenticação."""
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws = mock_ws_class.return_value
        mock_ws.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        mock_ws.connect.side_effect = Exception("ConnectFail")
        assert await orch._setup_session() is False
        mock_ws.connect.side_effect = None
        orch.auth.open_trading_session = AsyncMock(
            return_value=DerivTradingSession(ws_url="wss://test/ws?otp=x", balance=1000.0, account_id="DOT1")
        )
        assert await orch._setup_session() is True


@pytest.mark.asyncio
async def test_orchestrator_run_loop_reconnection(orchestrator_config):
    """Testa a reconexão automática no loop."""
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws = mock_ws_class.return_value
        mock_ws.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch.ws.is_running = False
        orch._setup_session = AsyncMock(return_value=True)
        orch._start_streams = AsyncMock(return_value=True)
        orch.running = True

        async def stop_soon(_):
            orch.running = False

        with patch("src.application.services.orchestrator.asyncio.sleep", side_effect=stop_soon):
            await asyncio.wait_for(orch.run(), timeout=2.0)


@pytest.mark.asyncio
async def test_orchestrator_stop(orchestrator_config):
    """Testa a parada graciosa."""
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch.running = True
        await orch.stop()
        assert orch.running is False


@pytest.mark.asyncio
async def test_orchestrator_setup_session_auth_error(orchestrator_config):
    """Cobre erro de autorização."""
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_cls:
        mock_ws_cls.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch.auth.open_trading_session = AsyncMock(side_effect=DerivRestError("token invalid"))
        assert await orch._setup_session() is False


@pytest.mark.asyncio
async def test_orchestrator_setup_session_generic_error(orchestrator_config):
    """Cobre excecao generica no setup (nao DerivRestError)."""
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_cls:
        mock_ws_cls.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch.auth.open_trading_session = AsyncMock(side_effect=RuntimeError("ws down"))
        assert await orch._setup_session() is False


@pytest.mark.asyncio
async def test_orchestrator_start_streams_exception(orchestrator_config):
    """Cobre exceção em streams."""
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch.stream.start_candle_stream = AsyncMock(side_effect=Exception("STREAM_FAIL"))
        assert await orch._start_streams() is False


@pytest.mark.asyncio
async def test_orchestrator_full_lifecycle_summary(orchestrator_config):
    """Cobre lifecycle ao liquidar cluster quando PnL ainda nao atinge stop-win."""
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws = mock_ws_class.return_value
        mock_ws.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch.running = True
        orch.risk_manager.initial_bankroll = 100.0
        c = Contract(
            contract_id=1,
            symbol="RDBULL",
            direction=TradeDirection.CALL,
            stake=10.0,
            payout=18.0,
            status=TradeStatus.OPEN,
            buy_price=10.0,
            proposal_id="p1",
            expiry_time=0,
        )
        await orch.state.add_contract(c)
        orch.risk_manager.active_contract_ids = [1]
        orch.risk_manager.contract_to_symbol[1] = "RDBULL"
        await orch._on_contract_update(
            {
                "proposal_open_contract": {
                    "contract_id": 1,
                    "is_settled": 1,
                    "status": "won",
                    "profit": 10,
                    "balance_after": 110,
                }
            }
        )
        assert orch.running is True


@pytest.mark.asyncio
async def test_orchestrator_run_early_return_when_setup_fails(orchestrator_config):
    """Cobre run() quando _setup_session falha antes do loop (linha 54)."""
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch._setup_session = AsyncMock(return_value=False)
        orch._start_streams = AsyncMock(return_value=True)
        await orch.run()
        assert orch.running is False


@pytest.mark.asyncio
async def test_orchestrator_run_early_return_when_streams_fail(orchestrator_config):
    """Cobre run() quando _start_streams falha na entrada (linha 54)."""
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch._setup_session = AsyncMock(return_value=True)
        orch._start_streams = AsyncMock(return_value=False)
        await orch.run()
        assert orch.running is False


@pytest.mark.asyncio
async def test_orchestrator_start_streams_success(orchestrator_config):
    """Cobre _start_streams sem exceção (await start_candle_stream, linha 101)."""
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch.stream.start_candle_stream = AsyncMock()
        assert await orch._start_streams() is True


@pytest.mark.asyncio
async def test_orchestrator_start_streams_retries_on_connection_error(orchestrator_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws = mock_ws_class.return_value
        mock_ws.subscribe = MagicMock()
        mock_ws.is_running = False
        orch = Orchestrator(orchestrator_config, "token")
        orch.stream.start_candle_stream = AsyncMock(side_effect=[ConnectionError("down"), None])
        with patch("src.application.services.orchestrator.asyncio.sleep", new_callable=AsyncMock):
            assert await orch._start_streams() is True
        assert mock_ws.connect.await_count == 1


@pytest.mark.asyncio
async def test_orchestrator_start_streams_fails_after_retry_limit(orchestrator_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws = mock_ws_class.return_value
        mock_ws.subscribe = MagicMock()
        mock_ws.is_running = False
        orch = Orchestrator(orchestrator_config, "token")
        orch.stream.start_candle_stream = AsyncMock(side_effect=ConnectionError("down"))
        with patch("src.application.services.orchestrator.asyncio.sleep", new_callable=AsyncMock):
            assert await orch._start_streams() is False
        assert mock_ws.connect.await_count == 1


@pytest.mark.asyncio
async def test_orchestrator_run_reconnect_fails_sleeps_five(orchestrator_config):
    """Cobre run() com ws inativo e reconexão falhando (await asyncio.sleep(5), linha 63)."""
    TradingState.reset()
    sleeps: list[float] = []

    async def track_sleep(delay: float) -> None:
        sleeps.append(delay)
        if delay == 5:
            orch.ws.is_running = True
        if len(sleeps) >= 12:
            orch.running = False

    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch.ws.is_running = False
        orch._setup_session = AsyncMock(side_effect=[True, False, True])
        orch._start_streams = AsyncMock(return_value=True)
        orch.running = True
        with patch("src.application.services.orchestrator.asyncio.sleep", side_effect=track_sleep):
            await asyncio.wait_for(orch.run(), timeout=5.0)
    assert 5 in sleeps


@pytest.mark.asyncio
async def test_orchestrator_run_loop_persistence_and_reconcile(orchestrator_config):
    """Cobre o corpo do while em run(): estado, persistência e reconcile (linhas 65-78)."""
    TradingState.reset()
    sleeps: list[float] = []

    async def stop_after_main_loops(delay: float) -> None:
        sleeps.append(delay)
        if delay == 1 and sleeps.count(1) >= 5:
            orch.running = False

    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        orch.ws.is_running = True
        orch._setup_session = AsyncMock(return_value=True)
        orch._start_streams = AsyncMock(return_value=True)
        orch.state.active_contracts[99] = Contract(
            contract_id=99,
            proposal_id="p99",
            status=TradeStatus.OPEN,
            buy_price=1.0,
            payout=2.0,
            symbol="RDBULL",
            direction=TradeDirection.CALL,
            stake=1.0,
            expiry_time=0,
        )
        orch.executor.reconcile = AsyncMock()
        orch.persistence.save = MagicMock()
        orch.running = True
        with patch("src.application.services.orchestrator.asyncio.sleep", side_effect=stop_after_main_loops):
            await asyncio.wait_for(orch.run(), timeout=5.0)
        orch.executor.reconcile.assert_called()
        orch.persistence.save.assert_called()

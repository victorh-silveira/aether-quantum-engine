"""Testes unitarios para o ciclo de vida do Orquestrador (motor podado)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.infrastructure.api.deriv_rest_client import DerivRestError, DerivTradingSession
from src.infrastructure.state.trading_state import TradingState
from tests.unit.application.orchestrator_session_patches import session_setup_patches


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
        mock_ws.ws = None
        orch = Orchestrator(orchestrator_config, "token")
        with session_setup_patches(otp_ok=False, public_side_effect=RuntimeError("HANDSHAKE_TIMEOUT")):
            assert await orch._setup_session() is False
        with session_setup_patches(otp_ok=True):
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
            patch(
                "src.application.services.orchestrator.orchestrator_run_loop.asyncio.sleep",
                side_effect=stop_soon,
            ),
        ):
            await asyncio.wait_for(orch.run(), timeout=2.0)


@pytest.mark.asyncio
async def test_orchestrator_setup_session_auth_error(orchestrator_config):
    """Cobre erro de autorização."""
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_cls:
        mock_ws_cls.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        with patch(
            "src.application.services.orchestrator.ws_bootstrap._resolve_rest_account_balance",
            AsyncMock(side_effect=DerivRestError("token invalid")),
        ):
            assert await orch._setup_session() is False


@pytest.mark.asyncio
async def test_orchestrator_setup_session_generic_error(orchestrator_config):
    """Cobre excecao generica no setup (nao DerivRestError)."""
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_cls:
        mock_ws_cls.return_value.subscribe = MagicMock()
        orch = Orchestrator(orchestrator_config, "token")
        with patch(
            "src.application.services.orchestrator.ws_bootstrap._resolve_rest_account_balance",
            AsyncMock(side_effect=RuntimeError("ws down")),
        ):
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
        orch.config.setdefault("risk_management", {})["large_account_stop_win_pct"] = 15.0
        orch.running = True
        orch.risk_manager.initial_bankroll = 100.0
        c = Contract(
            contract_id=1,
            symbol="R_10",
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
        orch.risk_manager.contract_to_symbol[1] = "R_10"
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
        orch.auth.open_trading_session = AsyncMock(
            return_value=DerivTradingSession(ws_url="wss://test/ws?otp=x", balance=1000.0, account_id="DOT1")
        )
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
        orch.auth.open_trading_session = AsyncMock(
            return_value=DerivTradingSession(ws_url="wss://test/ws?otp=x", balance=1000.0, account_id="DOT1")
        )
        orch.stream.start_candle_stream = AsyncMock(side_effect=ConnectionError("down"))
        with patch("src.application.services.orchestrator.asyncio.sleep", new_callable=AsyncMock):
            assert await orch._start_streams() is False
        assert mock_ws.connect.await_count == 1


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
            symbol="R_10",
            direction=TradeDirection.CALL,
            stake=1.0,
            expiry_time=0,
        )
        orch.executor.reconcile = AsyncMock()
        orch.persistence.save = MagicMock()
        orch.running = True
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
            patch(
                "src.application.services.orchestrator.orchestrator_run_loop.asyncio.sleep",
                side_effect=stop_after_main_loops,
            ),
        ):
            await asyncio.wait_for(orch.run(), timeout=5.0)
        orch.executor.reconcile.assert_called()
        orch.persistence.save.assert_called()

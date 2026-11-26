"""Testes unitarios para fechar 100% em ws_bootstrap e warm_up_buffer_guard."""

from __future__ import annotations

import asyncio
import urllib.error
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import (
    warm_up_buffer_guard as wug,
    ws_bootstrap as wb,
)
from src.domain.risk.risk_manager import RiskManager
from src.infrastructure.api.deriv_rest_client import DerivRestError
from src.infrastructure.state.trading_state import TradingState


def _orch():
    TradingState.reset()
    state = TradingState()
    return SimpleNamespace(
        config={"orchestrator": {"broker_handshake_timeout_seconds": 1.0}, "data_handler": {"history_bars": 50}},
        infra=SimpleNamespace(enabled=False),
        state=state,
        risk_manager=RiskManager({"params": {"stake_min": 1.0}}),
        ws=SimpleNamespace(ws=MagicMock(), is_running=True, close=AsyncMock(), subscribe=MagicMock()),
        stream=SimpleNamespace(config={}, start_candle_stream=AsyncMock(), tick_buffer=None),
        auth=MagicMock(),
        trade_handler=SimpleNamespace(deriv_account_id="", trading_transport="ws"),
        symbols=["R_10"],
        trading_transport="ws",
        deriv_account_id="",
        _on_contract_update=AsyncMock(),
        _on_candle=AsyncMock(),
        logger=MagicMock(),
    )


@pytest.mark.asyncio
async def test_warm_up_buffer_guard_all_branches():
    orch = _orch()
    assert wug._timing({"orchestrator": {"stream_warm_up_delay_seconds": 2.0}})["stream_warm_up_delay_seconds"] == 2.0
    assert wug.stream_warm_up_active(orch) is False
    assert wug.stream_warm_up_remaining(orch) == 0.0
    wug.log_warm_up_guard_suspension(orch)

    delay = wug.schedule_stream_warm_up_barrier(orch)
    assert delay >= 0.0
    if delay > 0.0:
        assert wug.stream_warm_up_active(orch) is True
        assert wug.stream_warm_up_remaining(orch) > 0.0
        assert wug.trading_cycle_warm_up_suspended(orch) == wug.SIGNAL_SUSPENDED
        wug.log_warm_up_guard_suspension(orch)
    orch._stream_warmed_up_at = 0.0
    assert wug.trading_cycle_warm_up_suspended(orch) is None

    # _tick_buffer_has_live_data
    assert wug._tick_buffer_has_live_data(orch) is False
    orch.stream.tick_buffer = SimpleNamespace(last_tick_monotonic=lambda: 0.0)
    assert wug._tick_buffer_has_live_data(orch) is False
    now = asyncio.get_running_loop().time()
    orch.stream.tick_buffer = SimpleNamespace(last_tick_monotonic=lambda: now)
    assert wug._tick_buffer_has_live_data(orch, now=now) is True

    # await_stream_warm_up_gate waiver on timeout
    orch._stream_warmed_up_at = now + 50.0
    orch.stream.tick_buffer = None
    assert await wug.await_stream_warm_up_gate(orch, timeout=0.0) is True
    assert orch._warm_up_waiver_applied is True


@pytest.mark.asyncio
async def test_ws_bootstrap_setup_failure_and_streams_branches():
    orch = _orch()
    assert wb._setup_trading_session_failure(orch, DerivRestError("rest")) is False
    assert wb._setup_trading_session_failure(orch, urllib.error.HTTPError("http://x", 502, "Bad", {}, None)) is False
    assert wb._setup_trading_session_failure(orch, RuntimeError("HANDSHAKE_TIMEOUT")) is False
    assert wb._setup_trading_session_failure(orch, ConnectionError("conn")) is False
    assert wb._setup_trading_session_failure(orch, RuntimeError("other")) is False
    assert wb._setup_trading_session_failure(orch, ValueError("val")) is False

    # setup_trading_session OTP fallback -> REST
    with (
        patch("src.application.services.orchestrator.ws_bootstrap.validate_infra_services", AsyncMock()),
        patch("src.application.services.orchestrator.ws_bootstrap.restore_orchestrator_state", AsyncMock()),
        patch(
            "src.application.services.orchestrator.ws_bootstrap._resolve_rest_account_balance",
            AsyncMock(return_value=("VRTC1", 500.0)),
        ),
        patch(
            "src.application.services.orchestrator.ws_bootstrap._broker_pat_websocket_handshake",
            AsyncMock(side_effect=TimeoutError("otp")),
        ),
        patch("src.application.services.orchestrator.ws_bootstrap.open_public_market_handshake", AsyncMock()),
        patch("src.application.services.orchestrator.ws_bootstrap.bootstrap_active_session_targets", AsyncMock()),
    ):
        assert await wb.setup_trading_session(orch) is True
        assert orch.trading_transport == "rest"

    # OTP success
    orch2 = _orch()
    with (
        patch("src.application.services.orchestrator.ws_bootstrap.validate_infra_services", AsyncMock()),
        patch("src.application.services.orchestrator.ws_bootstrap.restore_orchestrator_state", AsyncMock()),
        patch(
            "src.application.services.orchestrator.ws_bootstrap._resolve_rest_account_balance",
            AsyncMock(return_value=("VRTC1", 500.0)),
        ),
        patch(
            "src.application.services.orchestrator.ws_bootstrap._broker_pat_websocket_handshake",
            AsyncMock(return_value=SimpleNamespace(balance=500.0, account_id="VRTC1")),
        ),
        patch("src.application.services.orchestrator.ws_bootstrap.subscribe_account_transactions", AsyncMock()),
        patch("src.application.services.orchestrator.ws_bootstrap.bootstrap_active_session_targets", AsyncMock()),
    ):
        assert await wb.setup_trading_session(orch2) is True
        assert orch2.trading_transport == "ws"

    # start_orchestrator_streams retry + failure
    orch3 = _orch()
    orch3.ws.is_running = False
    orch3.stream.start_candle_stream = AsyncMock(side_effect=[ConnectionError("drop"), None])
    with (
        patch("src.application.services.orchestrator.ws_bootstrap.asyncio.sleep", AsyncMock()),
        patch(
            "src.application.services.orchestrator.ws_bootstrap.open_broker_handshake",
            AsyncMock(return_value=SimpleNamespace(balance=300.0)),
        ),
    ):
        assert await wb.start_orchestrator_streams(orch3) is True

    orch3.trading_transport = "rest"
    orch3.ws.is_running = False
    orch3.stream.start_candle_stream = AsyncMock(side_effect=[ConnectionError("drop"), ConnectionError("drop2")])
    with (
        patch("src.application.services.orchestrator.ws_bootstrap.asyncio.sleep", AsyncMock()),
        patch("src.application.services.orchestrator.ws_bootstrap.open_public_market_handshake", AsyncMock()),
    ):
        assert await wb.start_orchestrator_streams(orch3) is False

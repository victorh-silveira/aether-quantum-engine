import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.warm_up_buffer_guard import STREAM_WARM_UP_DELAY_SECONDS
from src.infrastructure.handlers.stream_reconnect import (
    _continuous_stream_active,
    _needs_profit_table_audit,
    execute_stream_reconnect,
)


def _build_reconnect_mocks(
    *,
    stream_ready_mono: float = 0.0,
    ws_url: str = "wss://test",
    refresh_side_effect: Exception | None = None,
    active_contracts: dict | None = None,
    symbols: list[str] | None = None,
    use_fallback_session: bool = False,
):
    orch = MagicMock()
    orch.running = True
    orch._stream_ready_mono = stream_ready_mono
    orch.config = {"orchestrator": {"stream_warm_up_delay_seconds": STREAM_WARM_UP_DELAY_SECONDS}}
    orch.risk_manager.pending_loss = {}
    orch.risk_manager.pending_loss_total = lambda: 0.0
    orch.last_data_signature = "sig"
    orch._signature_invalidation_logged_key = "sig"
    orch._last_processed_epoch = 1
    orch._quality_guard_logged_cycle_id = 0
    orch.logger = MagicMock()
    orch.ws.ws = MagicMock()
    orch.ws.close = AsyncMock()
    if use_fallback_session:
        del orch.auth.refresh_otp_ws_url
        orch.auth.open_trading_session = AsyncMock(
            return_value=MagicMock(ws_url=ws_url),
        )
    elif refresh_side_effect is not None:
        orch.auth.refresh_otp_ws_url = AsyncMock(side_effect=refresh_side_effect)
    else:
        orch.auth.refresh_otp_ws_url = AsyncMock(return_value=ws_url)
    orch.ws.connect = AsyncMock()
    orch._on_contract_update = MagicMock()
    orch._on_transaction = MagicMock()
    orch.state.active_contracts = active_contracts or {}
    orch.risk_manager.active_contract_ids = list(active_contracts or {})
    orch._reconciliation_pending = False
    orch.trade_handler = MagicMock()

    stream = MagicMock()
    stream.candle_callback = AsyncMock()
    stream.symbols = symbols or ["R_10"]
    stream.macro_granularity = 900
    stream.micro_granularity = 300
    stream.ws = orch.ws
    stream.ws.send = AsyncMock()
    stream._on_candle = MagicMock()
    stream._on_tick = MagicMock()
    stream.tick_buffer = MagicMock()
    stream.tick_buffer.reset_live_accumulators = MagicMock()
    stream.tick_buffer.touch_activity = MagicMock()
    orch.config = {"orchestrator": {"stream_warm_up_delay_seconds": 45}}
    return orch, stream


@contextmanager
def _reconnect_patches():
    with (
        patch(
            "src.infrastructure.handlers.stream_reconnect.ws_connect_options",
            return_value={"max_attempts": 1},
        ),
        patch(
            "src.infrastructure.handlers.stream_reconnect.subscribe_account_transactions",
            new_callable=AsyncMock,
        ),
    ):
        yield


@pytest.mark.asyncio
async def test_execute_stream_reconnect_fallback_open_trading_session():
    orch, stream = _build_reconnect_mocks(ws_url="wss://legacy-otp", use_fallback_session=True)
    with _reconnect_patches():
        ok = await execute_stream_reconnect(orch, stream)
    assert ok is True
    orch.auth.open_trading_session.assert_awaited_once()
    orch.ws.connect.assert_awaited_once_with("wss://legacy-otp", max_attempts=1)


@pytest.mark.asyncio
async def test_execute_stream_reconnect_refreshes_otp_before_connect():
    orch, stream = _build_reconnect_mocks(ws_url="wss://fresh-otp")
    with _reconnect_patches():
        ok = await execute_stream_reconnect(orch, stream)
    assert ok is True
    orch.auth.refresh_otp_ws_url.assert_awaited_once()
    orch.ws.connect.assert_awaited_once_with("wss://fresh-otp", max_attempts=1)


@pytest.mark.asyncio
async def test_execute_stream_reconnect_success():
    orch, stream = _build_reconnect_mocks()
    with _reconnect_patches():
        ok = await execute_stream_reconnect(orch, stream)
    assert ok is True
    orch.ws.close.assert_awaited_once()
    stream.ws.send.assert_awaited()
    stream.tick_buffer.reset_live_accumulators.assert_called_once()
    stream.tick_buffer.touch_activity.assert_called_once()
    assert orch._stream_warmed_up_at > 0.0


@pytest.mark.asyncio
async def test_execute_stream_reconnect_schedules_warm_up_barrier():
    orch, stream = _build_reconnect_mocks()
    with _reconnect_patches():
        ok = await execute_stream_reconnect(orch, stream)
    assert ok is True
    assert orch._stream_warmed_up_at - asyncio.get_running_loop().time() == pytest.approx(
        STREAM_WARM_UP_DELAY_SECONDS,
        abs=0.05,
    )


@pytest.mark.asyncio
async def test_execute_stream_reconnect_skips_ohlc_on_continuous_session():
    orch, stream = _build_reconnect_mocks(stream_ready_mono=999.0, symbols=["R_10", "R_50"])
    with _reconnect_patches():
        ok = await execute_stream_reconnect(orch, stream)
    assert ok is True
    stream.ws.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_stream_reconnect_schedules_profit_audit_on_failure():
    orch, stream = _build_reconnect_mocks(
        refresh_side_effect=ConnectionError("broker indisponivel"),
    )
    with patch(
        "src.infrastructure.handlers.stream_reconnect.ws_connect_options",
        return_value={"max_attempts": 1},
    ):
        ok = await execute_stream_reconnect(orch, stream)
    assert ok is False
    orch.trade_handler.schedule_profit_table_audit.assert_called_once_with(
        orch,
        reason="broker_unavailable",
    )


@pytest.mark.asyncio
async def test_execute_stream_reconnect_schedules_profit_audit_with_active_contracts():
    orch, stream = _build_reconnect_mocks(
        stream_ready_mono=50.0,
        active_contracts={101: MagicMock()},
    )
    orch.risk_manager.active_contract_ids = [101]
    with _reconnect_patches():
        ok = await execute_stream_reconnect(orch, stream)
    assert ok is True
    orch.trade_handler.schedule_profit_table_audit.assert_called_once_with(
        orch,
        reason="stream_reconnect",
    )


@pytest.mark.asyncio
async def test_execute_stream_reconnect_without_callback():
    orch = MagicMock()
    stream = MagicMock()
    stream.candle_callback = None
    assert await execute_stream_reconnect(orch, stream) is False


@pytest.mark.asyncio
async def test_execute_stream_reconnect_failure():
    orch, stream = _build_reconnect_mocks(refresh_side_effect=RuntimeError("ws fail"))
    with patch(
        "src.infrastructure.handlers.stream_reconnect.ws_connect_options",
        return_value={"max_attempts": 1},
    ):
        assert await execute_stream_reconnect(orch, stream) is False


def test_continuous_stream_active_requires_running():
    orch = MagicMock()
    orch._stream_ready_mono = 10.0
    orch.running = False
    assert _continuous_stream_active(orch) is False
    orch.running = True
    assert _continuous_stream_active(orch) is True


def test_needs_profit_table_audit_active_contracts():
    orch = MagicMock()
    orch.state.active_contracts = {1: object()}
    orch.risk_manager.active_contract_ids = []
    orch._reconciliation_pending = False
    assert _needs_profit_table_audit(orch) is True


def test_needs_profit_table_audit_risk_ids_only():
    orch = MagicMock()
    orch.state.active_contracts = {}
    orch.risk_manager.active_contract_ids = [42]
    orch._reconciliation_pending = False
    assert _needs_profit_table_audit(orch) is True


def test_needs_profit_table_audit_reconciliation_pending():
    orch = MagicMock()
    orch.state.active_contracts = {}
    orch.risk_manager.active_contract_ids = []
    orch._reconciliation_pending = True
    assert _needs_profit_table_audit(orch) is True

"""Fixtures compartilhadas para testes unitarios da camada de aplicacao."""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.infrastructure.state.trading_state import TradingState
from tests.market_symbols import ALL_SYMBOLS, ANCHOR


@pytest.fixture
def orch_config():
    """Configuracao minima do orquestrador Laya continuo em R_10 M5."""
    return {
        "anchor": ANCHOR,
        "symbols": list(ALL_SYMBOLS),
        "api_config": {
            "request_timeout_seconds": 30,
        },
        "stream": {
            "micro_granularity": 300,
            "ticks_limit": 200,
            "buffer_size": 300,
        },
        "laya": {
            "enabled": True,
            "model_id": "convaiinnovations/laya",
            "device": "cpu",
            "context_candles": 12,
        },
        "risk_management": {
            "params": {
                "duration": 5,
                "duration_unit": "m",
                "payout_estimate": 0.95,
                "stake_min": 1.0,
                "min_stake_pct": 0.01,
                "max_stake_pct": 0.05,
            },
        },
        "orchestrator": {
            "engine_mode": "execute",
            "reconcile_interval_seconds": 1,
            "cycle_interval_seconds": 0,
            "require_signature_boundary": False,
            "post_settlement_breath_seconds": 0,
            "execution": {
                "include_anchor_trades": True,
                "inter_symbol_delay": 0.0,
            },
        },
        "infra": {"enabled": False},
        "trading": {"mode": "demo", "session": {"enabled": False}},
    }


@pytest.fixture
async def orch_ready(orch_config):
    """Instancia pronta do Orchestrator com WebSocket mockado."""
    TradingState.reset()
    ws_patch = patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock())
    mock_ws_class = ws_patch.start()
    mock_ws = mock_ws_class.return_value
    mock_ws.subscribe = MagicMock()
    orch = Orchestrator(orch_config, "token")
    orch.stream.is_synchronized = True
    orch.ws.is_running = True
    orch.running = True
    orch.state.balance = 1000.0
    orch.risk_manager.set_initial_bankroll(1000.0)
    orch._stream_ready_at = 0.0
    yield orch
    pending = orch._post_settlement_task
    if isinstance(pending, asyncio.Task) and not pending.done():
        pending.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pending
    worker = getattr(orch, "_settlement_worker_task", None)
    if isinstance(worker, asyncio.Task) and not worker.done():
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker
    ws_patch.stop()

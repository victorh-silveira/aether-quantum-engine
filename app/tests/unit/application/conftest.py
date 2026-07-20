import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.engine_mode import ENGINE_MODE_TRAIN, apply_engine_mode
from src.infrastructure.state.trading_state import TradingState
from tests.market_symbols import ALL_SYMBOLS, ANCHOR


@pytest.fixture(autouse=True)
def fast_bootstrap_sleep():
    """Evita asyncio.sleep real no bootstrap durante a suíte de testes."""
    with patch(
        "src.application.services.deep_learning.dl_bootstrap_train.asyncio.sleep",
        new_callable=AsyncMock,
    ):
        yield


@pytest.fixture
def orch_config():
    return {
        "api_config": {
            "rest_base_url": "https://api.derivws.com",
            "deriv_app_id": "test-app-id",
            "public_ws_url": "ws://test",
            "request_timeout_seconds": 1,
        },
        "symbols": list(ALL_SYMBOLS),
        "anchor": ANCHOR,
        "deep_learning": {
            "enabled": True,
            "confidence_call_threshold": 0.75,
            "confidence_put_threshold": 0.25,
            "min_val_accuracy": 0.53,
            "lookback": 30,
            "training_history_bars": 120,
            "validation_bars": 10,
            "training_epochs": 2,
        },
        "data_handler": {"fetch_count": 100, "min_required_points": 2, "buffer_limit": 1000},
        "strategy": {
            "clusters": {"rd": [ANCHOR]},
            "correlation": {"anchor": ANCHOR},
        },
        "risk_management": {
            "small_account_threshold": 100.0,
            "small_account_stake": 1.0,
            "small_account_stop_win": 10.0,
            "large_account_stake_pct": 2.0,
            "large_account_stop_win_pct": 1.0,
            "params": {
                "duration": 2,
                "duration_unit": "m",
                "payout_estimate": 0.95,
                "entry_cooldown_ticks": 60,
                "stake_min": 0.5,
                "base_stake_min_pct": 0.01,
                "base_stake_max_pct": 0.02,
            },
            "kelly": {"fraction": 0.5, "base_win_rate": 0.55},
        },
        "orchestrator": {
            "engine_mode": "execute",
            "reconcile_interval_seconds": 1,
            "cycle_interval_seconds": 0,
            "execution": {
                "include_anchor_trades": True,
                "inter_symbol_delay": 0.25,
                "regime_evaluator": {"enabled": True},
            },
        },
        "infra": {"enabled": False},
        "trading": {"mode": "demo", "session": {"enabled": False}},
    }


@pytest.fixture
def orch_config_train(orch_config):
    apply_engine_mode(orch_config, ENGINE_MODE_TRAIN)
    return orch_config


@pytest.fixture
async def orch_ready(orch_config):
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
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    yield orch
    deferred = getattr(orch, "_dl_deferred_tasks", None)
    if isinstance(deferred, dict):
        for task in list(deferred.values()):
            if isinstance(task, asyncio.Task) and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        deferred.clear()
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


@pytest.fixture
async def orch_ready_train(orch_config_train):
    TradingState.reset()
    ws_patch = patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock())
    mock_ws_class = ws_patch.start()
    mock_ws = mock_ws_class.return_value
    mock_ws.subscribe = MagicMock()
    orch = Orchestrator(orch_config_train, "token")
    orch.stream.is_synchronized = True
    orch.ws.is_running = True
    orch.running = True
    orch.state.balance = 1000.0
    orch.risk_manager.set_initial_bankroll(1000.0)
    orch._stream_ready_at = 0.0
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    yield orch
    deferred = getattr(orch, "_dl_deferred_tasks", None)
    if isinstance(deferred, dict):
        for task in list(deferred.values()):
            if isinstance(task, asyncio.Task) and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        deferred.clear()
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

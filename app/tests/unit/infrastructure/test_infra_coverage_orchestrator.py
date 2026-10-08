"""Cobertura de orquestrador, squeeze e restore de estado."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.deep_learning.dl_model_artifacts import schedule_model_upload
from src.application.services.execution_volatility_threshold import resolve_dynamic_threshold_bundle
from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.orchestrator_persistence import save_full_state
from src.application.services.orchestrator.orchestrator_state_restore import (
    bar_epoch_already_processed,
    mark_bar_processed,
    restore_orchestrator_state,
)
from src.domain.math.probability_entropy import entropy_penalty_factor


def test_schedule_model_upload_without_loop(tmp_path):
    orch = MagicMock()
    orch.infra = MagicMock(enabled=True)
    path = tmp_path / "m.pth"
    path.write_bytes(b"x")
    with patch("asyncio.get_running_loop", side_effect=RuntimeError), patch("asyncio.run") as run_mock:
        schedule_model_upload(orch, "R_10", path, arch="tcn")
    run_mock.assert_called_once()


@pytest.mark.asyncio
async def test_save_full_state_writes_legacy_persistence(tmp_path):
    config = {
        "api_config": {"request_timeout_seconds": 1},
        "symbols": ["R_10"],
        "data_handler": {},
        "risk_management": {"params": {}, "kelly": {}, "limits": {}},
        "orchestrator": {},
        "infra": {"enabled": False},
    }
    orch = Orchestrator(config, MagicMock())
    orch.state.get_state = AsyncMock(return_value={"balance": 1.0})
    orch.risk_manager.get_state = MagicMock(return_value={})
    orch.get_data_state_signature = MagicMock(return_value="sig-1")
    await save_full_state(orch)
    assert orch.persistence.load() is not None


@pytest.mark.asyncio
async def test_save_full_state_legacy_store_without_bundle():
    class LegacyStore:
        save_snapshot = AsyncMock()

    orch = MagicMock()
    orch.state.get_state = AsyncMock(return_value={"balance": 1.0})
    orch.risk_manager.get_state = MagicMock(return_value={})
    orch.risk_manager.total_session_profit = 0.0
    orch.get_data_state_signature = MagicMock(return_value="sig-legacy")
    store = LegacyStore()
    orch.state_store = store
    orch.persistence = MagicMock()
    with (
        patch(
            "src.application.services.orchestrator.orchestrator_persistence.persist_session_hash",
            AsyncMock(),
        ) as persist_mock,
        patch(
            "src.application.services.orchestrator.orchestrator_persistence.sync_market_signature",
            AsyncMock(),
        ) as sync_mock,
    ):
        await save_full_state(orch)
    store.save_snapshot.assert_awaited_once()
    persist_mock.assert_awaited_once()
    sync_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_orchestrator_skips_processed_bar():
    config = {
        "api_config": {"request_timeout_seconds": 1},
        "symbols": ["R_10"],
        "anchor": "R_10",
        "data_handler": {},
        "risk_management": {"params": {}, "kelly": {}, "limits": {}},
        "orchestrator": {},
        "infra": {"enabled": False},
    }
    orch = Orchestrator(config, MagicMock())
    candle = MagicMock(symbol=orch.anchor, epoch=99)
    with patch(
        "src.application.services.orchestrator.bar_epoch_already_processed",
        AsyncMock(return_value=True),
    ):
        await orch._on_candle(candle)
    assert orch.tick_count == 0


@pytest.mark.asyncio
async def test_bar_epoch_missing_stored_value():
    orch = MagicMock()
    orch.infra = MagicMock(enabled=True)
    orch.state_store = AsyncMock()
    orch.state_store.get_string.return_value = ""
    assert await bar_epoch_already_processed(orch, "R_10", 1) is False


def test_entropy_penalty_below_floor():
    assert entropy_penalty_factor(0.99, ceiling=0.92, floor=0.8) == 0.0


def test_entropy_penalty_at_ceiling():
    assert entropy_penalty_factor(0.5, ceiling=0.01, floor=0.0) == 1.0


@pytest.mark.asyncio
async def test_restore_orchestrator_no_store():
    orch = MagicMock()
    orch.state_store = None
    await restore_orchestrator_state(orch)


@pytest.mark.asyncio
async def test_mark_bar_processed_no_store():
    orch = MagicMock()
    orch.state_store = None
    await mark_bar_processed(orch, "R_10", 1)


def test_squeeze_dynamic_threshold_bundle():
    bundle = resolve_dynamic_threshold_bundle(
        base_call=0.53,
        base_put=0.47,
        base_edge=0.04,
        bb_width=0.001,
        atr_norm=0.01,
        adx=0.1,
        vol_ratio=0.8,
        bb_width_history=[0.05, 0.04],
        symbol="R_10",
        implied_vol_ratio=0.25,
        cfg={
            "enabled": True,
            "squeeze_edge_exponential_k": 2.5,
            "implied_vol_bb_scale": True,
        },
    )
    assert bundle is not None
    assert bundle.min_edge >= 0.04

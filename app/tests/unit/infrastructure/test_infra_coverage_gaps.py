"""Cobertura de ramos da infraestrutura e modulos auxiliares."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.deep_learning.dl_calibration import CalibratorState
from src.application.services.deep_learning.dl_calibration_fit import (
    calibrator_entropy_metrics,
    entropy_weight_penalty,
)
from src.application.services.deep_learning.dl_model_artifacts import (
    ensure_local_model_checkpoint,
    schedule_model_upload,
    upload_all_symbol_checkpoints,
    upload_model_checkpoint,
)
from src.application.services.execution_direction_resolver import _dl_call_put_scores
from src.application.services.execution_squeeze_bias import normalize_bias_side
from src.application.services.execution_squeeze_gate import passes_squeeze_gate
from src.application.services.execution_volatility_bb import _percentile_p10
from src.application.services.orchestrator.orchestrator_state_restore import (
    bar_epoch_already_processed,
    persist_session_hash,
    sync_market_signature,
)
from src.domain.math.probability_entropy import entropy_penalty_factor
from src.domain.risk.risk_manager_restore import apply_risk_snapshot
from src.infrastructure.factories.infra_factory import validate_infra_services
from src.infrastructure.state.json_state_store import JsonStateStore
from src.infrastructure.state.redis_state_store import RedisStateStore


def test_normalize_bias_side_variants():
    assert normalize_bias_side("RISE") == "call"
    assert normalize_bias_side("FALL") == "put"
    assert normalize_bias_side("flat") is None
    assert normalize_bias_side(0.51) is None
    assert normalize_bias_side(0.6) == "call"
    assert normalize_bias_side(0.4) == "put"


def test_squeeze_gate_resolved_mismatch():
    metrics = {
        "squeeze_extreme": True,
        "direction_margin": 0.2,
        "trend_direction": "CALL",
        "indicator_regime_side": "call",
        "calibrated_prob": 0.7,
        "resolved_direction": "PUT",
    }
    assert passes_squeeze_gate(metrics, cfg={"squeeze_min_margin": 0.12}) is False


def test_percentile_p10_single_value():
    assert _percentile_p10([0.3]) == 0.3
    assert _percentile_p10([]) == 0.0


def test_entropy_penalty_mid_range():
    assert 0.0 < entropy_penalty_factor(0.7, ceiling=1.0, floor=0.5) < 1.0


def test_dl_call_put_entropy_violation():
    entry = {
        "metrics": {
            "calibrated_prob": 0.6,
            "entropy_violation": True,
            "dynamic_call_threshold": 0.53,
            "dynamic_put_threshold": 0.47,
        }
    }
    call, put = _dl_call_put_scores(entry, {"dl_raw_weight": 0.45})
    assert call != put


def test_calibrator_entropy_empty_probs():
    meta = calibrator_entropy_metrics([], [], CalibratorState())
    assert meta["calibrated_entropy"] == 0.0


def test_entropy_weight_penalty_callable():
    assert entropy_weight_penalty(0.6) >= 0.0


def test_apply_risk_snapshot_loss_fields():
    mgr = MagicMock()
    apply_risk_snapshot(
        mgr,
        {"last_loss_symbol": "R_10", "last_loss_direction": "PUT"},
    )
    assert mgr.last_loss_symbol == "R_10"


@pytest.mark.asyncio
async def test_json_state_store_load_errors(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    store = JsonStateStore(path)
    assert await store.load_snapshot() is None
    assert store.load() is None
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    store2 = JsonStateStore(empty)
    assert store2.load() is None


@pytest.mark.asyncio
async def test_redis_load_invalid_json():
    client = AsyncMock()
    client.get.return_value = "not-json"
    store = RedisStateStore(url="redis://localhost/0", debounce_seconds=0.0)
    with patch.object(store, "_redis", AsyncMock(return_value=client)):
        assert await store.load_snapshot() is None


@pytest.mark.asyncio
async def test_redis_close_flushes():
    client = AsyncMock()
    store = RedisStateStore(url="redis://localhost/0", debounce_seconds=1.0)
    store._pending_snapshot = {"a": 1}
    with patch.object(store, "_redis", AsyncMock(return_value=client)):
        await store.close()
    client.set.assert_called()


@pytest.mark.asyncio
async def test_persist_and_sync_helpers():
    orch = MagicMock()
    orch.state_store = None
    await persist_session_hash(orch)
    await sync_market_signature(orch, "")
    orch.state_store = AsyncMock()
    orch.state_mgr.state = MagicMock(
        initial_balance=1.0,
        current_balance=2.0,
        daily_stop_win_target=3.0,
        total_trades_today=1,
        stop_win_triggered=False,
        day_key=1,
    )
    await persist_session_hash(orch)
    await sync_market_signature(orch, "sig")


@pytest.mark.asyncio
async def test_bar_epoch_invalid_stored_value():
    orch = MagicMock()
    orch.infra = MagicMock(enabled=True)
    orch.state_store = AsyncMock()
    orch.state_store.get_string.return_value = "bad"
    assert await bar_epoch_already_processed(orch, "R_10", 1) is False


@pytest.mark.asyncio
async def test_validate_infra_warn_without_fail_fast():
    services = MagicMock(enabled=True, fail_fast=True)
    services.state_store.ping = AsyncMock(return_value=False)
    services.market_writer.ping = AsyncMock(return_value=True)
    services.model_store.head = AsyncMock(return_value=True)
    await validate_infra_services(services, {"infra": {"enabled": True, "fail_fast": False}})


@pytest.mark.asyncio
async def test_validate_infra_all_ok_logs():
    services = MagicMock(enabled=True, fail_fast=True)
    services.state_store.ping = AsyncMock(return_value=True)
    services.market_writer.ping = AsyncMock(return_value=True)
    services.model_store.head = AsyncMock(return_value=True)
    await validate_infra_services(services, {"infra": {"enabled": True}})


@pytest.mark.asyncio
async def test_model_artifacts_upload_and_schedule(tmp_path):
    orch = MagicMock()
    orch.infra = MagicMock(enabled=True)
    orch.model_store.upload = AsyncMock()
    path = tmp_path / "m.pth"
    path.write_bytes(b"x")
    await upload_model_checkpoint(orch, "R_10", path, arch="tcn")
    schedule_model_upload(orch, "R_10", path, arch="tcn")
    orch.infra.enabled = False
    schedule_model_upload(orch, "R_10", path, arch="tcn")


@pytest.mark.asyncio
async def test_ensure_local_and_upload_all(tmp_path):
    orch = MagicMock()
    orch.infra = MagicMock(enabled=True)
    orch.model_store.download_latest = AsyncMock(return_value=True)
    orch.model_store.upload = AsyncMock()
    orch.symbols = ["R_10"]
    orch.config = {
        "deep_learning": {"model_path_template": str(tmp_path / "{symbol}.pth"), "arch": "tcn"},
        "data_handler": {},
        "risk_management": {"params": {}},
    }
    await ensure_local_model_checkpoint(orch, "R_10", orch.config["deep_learning"], {"arch": "tcn"})
    file_path = tmp_path / "R_10.pth"
    file_path.write_bytes(b"z")
    await upload_all_symbol_checkpoints(orch)

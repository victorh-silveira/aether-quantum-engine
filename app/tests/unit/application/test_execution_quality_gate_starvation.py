import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.execution_quality_gate import (
    resolve_dynamic_quality_limits,
    starvation_decay_factor,
)
from src.application.services.execution_quality_gate_cluster import (
    _strongly_negative_meta,
    quality_conviction_suspends_cluster,
)
from src.application.services.execution_quality_gate_meta import evaluate_meta_payoff_quality
from src.application.services.execution_quality_gate_starvation import (
    REDIS_SKIPPED_CYCLES_COUNTER_KEY,
    _schedule_quality_skipped_cycles_persist,
    apply_starvation_edge_decay,
    apply_starvation_margin_decay,
    load_quality_skipped_cycles_counter,
    prepare_quality_skipped_cycles_counter,
    record_quality_guard_cycle_skip,
    reset_quality_skipped_cycles_counter,
    reset_quality_skipped_cycles_counter_for_orch,
)


def test_starvation_decay_factor_below_threshold_is_neutral():
    assert starvation_decay_factor(0) == 1.0
    assert starvation_decay_factor(2) == 1.0


def test_starvation_decay_factor_linear_decay_and_floor():
    assert starvation_decay_factor(6) == pytest.approx(0.90)
    assert starvation_decay_factor(9) == pytest.approx(0.60)
    assert starvation_decay_factor(13) == pytest.approx(0.20)
    assert starvation_decay_factor(20) == pytest.approx(0.20)


def test_apply_starvation_margin_decay_without_orch():
    margin, decay = apply_starvation_margin_decay(0.11, 9)
    assert decay == pytest.approx(0.60)
    assert margin == pytest.approx(0.066)


def test_apply_starvation_edge_decay_checks():
    assert apply_starvation_edge_decay(0.04, 2) == pytest.approx(0.04)
    assert apply_starvation_edge_decay(0.04, 9) == pytest.approx(0.01)


def test_apply_starvation_margin_decay_emits_deduped_log(orch_ready, caplog):
    orch = orch_ready
    with caplog.at_level("INFO", logger="AETH"):
        margin, decay = apply_starvation_margin_decay(0.11, 9, orch=orch)
        apply_starvation_margin_decay(0.11, 9, orch=orch)
    assert decay == pytest.approx(0.60)
    assert margin == pytest.approx(0.066)
    escape_logs = [record for record in caplog.records if "Válvula de inanição ativa" in record.message]
    assert len(escape_logs) == 1
    assert "0.0660" in escape_logs[0].message
    assert "skipped_cycles=9" in escape_logs[0].message


def test_resolve_dynamic_quality_limits_applies_starvation_decay_at_counter_6():
    risk_manager = SimpleNamespace(dlambert_unit=16.0)
    limits = resolve_dynamic_quality_limits(
        {},
        risk_manager=risk_manager,
        linear=1,
        pending_loss_total=13.333333333333334,
        skipped_cycles_counter=9,
    )
    assert limits["min_direction_margin"] == pytest.approx(0.066)
    assert limits["min_payoff_edge"] == pytest.approx(0.01)
    assert limits["starvation_decay_factor"] == pytest.approx(0.60)
    assert limits["skipped_cycles_counter"] == pytest.approx(9.0)


def test_passes_execution_quality_starvation_allows_margin_008_after_decay():
    metrics = {
        "calibrated_prob": 0.58,
        "predicted_payoff_edge": 0.02,
        "meta_classifier_applied": True,
        "meta_payoff_edge_zscore": 0.55,
    }
    risk_manager = SimpleNamespace(
        consecutive_losses_linear=1,
        dlambert_unit=16.0,
        pending_loss_total=lambda: 13.333333333333334,
    )
    orch = SimpleNamespace(_quality_skipped_cycles_counter=9, logger=MagicMock())

    assert evaluate_meta_payoff_quality(metrics, risk_manager=risk_manager, orch=orch) is True
    assert metrics["quality_min_direction_margin"] == pytest.approx(0.066)
    assert metrics["quality_starvation_decay_factor"] == pytest.approx(0.60)


def test_starvation_decay_inactive_before_threshold():
    assert starvation_decay_factor(2) == 1.0
    margin, decay = apply_starvation_margin_decay(0.11, 2)
    assert decay == 1.0
    assert margin == pytest.approx(0.11)


def test_passes_execution_quality_keeps_flow_without_starvation():
    metrics = {
        "calibrated_prob": 0.58,
        "predicted_payoff_edge": 0.06,
        "meta_classifier_applied": True,
        "meta_payoff_edge_zscore": 0.55,
    }
    risk_manager = SimpleNamespace(
        consecutive_losses_linear=1,
        dlambert_unit=16.0,
        pending_loss_total=lambda: 13.333333333333334,
    )
    assert evaluate_meta_payoff_quality(metrics, risk_manager=risk_manager, skipped_cycles_counter=0) is True
    assert metrics["quality_min_direction_margin"] == pytest.approx(0.11)


def test_quality_conviction_suspends_cluster_does_not_increment_skipped_counter(orch_ready):
    orch = orch_ready
    orch._quality_skipped_cycles_counter = 4
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = lambda: 0.0
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = False
    decisions = {
        "RDBULL": {
            "metrics": {
                "calibrated_prob": 0.51,
                "predicted_payoff_edge": 0.01,
                "meta_classifier_applied": True,
                "meta_payoff_edge_zscore": 0.10,
                "edge_zscore_samples": 15,
                "deploy_ok": True,
                "direction": "CALL",
            }
        },
    }
    assert quality_conviction_suspends_cluster(orch, decisions) is True
    assert orch._quality_skipped_cycles_counter == 4


@pytest.mark.asyncio
async def test_quality_skipped_cycles_counter_redis_roundtrip():
    store = MagicMock()
    store.get_string = AsyncMock(return_value=None)
    store.set_string = AsyncMock()
    assert await load_quality_skipped_cycles_counter(store) == 0
    store.get_string = AsyncMock(return_value="7")
    assert await load_quality_skipped_cycles_counter(store) == 7
    await reset_quality_skipped_cycles_counter(store)
    store.set_string.assert_awaited_with(REDIS_SKIPPED_CYCLES_COUNTER_KEY, "0")


@pytest.mark.asyncio
async def test_quality_skipped_cycles_counter_invalid_string():
    store = MagicMock()
    store.get_string = AsyncMock(return_value="bad")
    assert await load_quality_skipped_cycles_counter(store) == 0


@pytest.mark.asyncio
async def test_prepare_quality_skipped_cycles_counter_loads_from_store():
    store = MagicMock()
    store.get_string = AsyncMock(return_value="12")
    orch = SimpleNamespace(state_store=store, _quality_skipped_cycles_counter=0)
    count = await prepare_quality_skipped_cycles_counter(orch)
    assert count == 12
    assert orch._quality_skipped_cycles_counter == 12


@pytest.mark.asyncio
async def test_reset_quality_skipped_cycles_counter_for_orch():
    store = MagicMock()
    store.set_string = AsyncMock()
    orch = SimpleNamespace(state_store=store, _quality_skipped_cycles_counter=9)
    await reset_quality_skipped_cycles_counter_for_orch(orch)
    store.set_string.assert_awaited_with(REDIS_SKIPPED_CYCLES_COUNTER_KEY, "0")
    assert orch._quality_skipped_cycles_counter == 0


@pytest.mark.asyncio
async def test_quality_skipped_cycles_counter_without_store():
    assert await load_quality_skipped_cycles_counter(None) == 0
    await reset_quality_skipped_cycles_counter(None)
    orch = SimpleNamespace(state_store=None, _quality_skipped_cycles_counter=0)
    assert await prepare_quality_skipped_cycles_counter(orch) == 0


def test_record_quality_guard_cycle_skip_without_running_loop():
    store = MagicMock()
    store.set_string = AsyncMock()
    orch = SimpleNamespace(state_store=store, _quality_skipped_cycles_counter=2)
    assert record_quality_guard_cycle_skip(orch) == 3
    assert orch._quality_skipped_cycles_counter == 3


@pytest.mark.asyncio
async def test_record_quality_guard_cycle_skip_persists_when_loop_active():
    store = MagicMock()
    store.set_string = AsyncMock()
    orch = SimpleNamespace(state_store=store, _quality_skipped_cycles_counter=6)
    assert record_quality_guard_cycle_skip(orch) == 7
    pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending)
    store.set_string.assert_awaited_with(REDIS_SKIPPED_CYCLES_COUNTER_KEY, "7")


def test_schedule_persist_raises_runtime_error_when_no_loop():
    orch = SimpleNamespace(state_store=MagicMock(), _quality_skipped_cycles_counter=3)
    with patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")):
        _schedule_quality_skipped_cycles_persist(orch)


def test_strongly_negative_meta_relaxed_under_starvation():
    metrics = {"meta_payoff_edge_zscore": -0.86}
    assert _strongly_negative_meta(metrics, decay_factor=1.0) is True
    assert _strongly_negative_meta(metrics, decay_factor=0.20) is False

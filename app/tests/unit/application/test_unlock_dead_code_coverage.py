from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.execution_direction_checks import (
    initial_direction_checks,
    meta_zscore_soft_ok,
)
from src.application.services.execution_quality_gate import apply_quality_penalty_to_metrics
from src.application.services.execution_quality_gate_cluster import quality_conviction_suspends_cluster
from src.application.services.execution_sniper_gates import hurst_regime_allowed
from src.application.services.meta_payoff_regression import apply_meta_regression_edge
from src.application.services.orchestrator.execution_blockers import _candidate_block_reason
from src.application.services.orchestrator.execution_collect import (
    collect_cluster_orders,
    mandatory_fallback_if_empty,
    resolve_mandatory_ultimate_candidate,
)
from src.application.services.orchestrator.regime_freeze_yield import propagate_cluster_signal_suspended
from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready
from src.application.services.regime_micro_freeze import SIGNAL_SUSPENDED
from src.domain.models.trade import TradeDirection
from src.domain.risk.soft_recovery_policy import risk_session_bankroll_pending


def test_meta_zscore_soft_ok_uses_floor():
    assert meta_zscore_soft_ok({"meta_payoff_edge_zscore": 0.0, "edge_zscore": 0.0}) is True


def test_initial_direction_checks_blocks_neutral_clamp():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "gate_reason": "neutral_clamp",
            "calibration_mode": "neutral_clamp",
            "calibrated_prob": 0.50,
            "raw_prob": 0.50,
            "execute": True,
            "deploy_ok": True,
        },
    }
    result = initial_direction_checks(entry, {})
    assert result is None
    assert entry["metrics"].get("gate_reason") == "neutral_clamp"
    assert entry["metrics"].get("quality_guard_reject") is True


def test_initial_direction_checks_clears_neutral_clamp_when_force_trade():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "gate_reason": "neutral_clamp",
            "calibration_mode": "neutral_clamp",
            "calibrated_prob": 0.50,
            "raw_prob": 0.50,
            "execute": True,
            "deploy_ok": True,
        },
    }
    result = initial_direction_checks(entry, {"force_trade_every_cycle": True})
    assert result is not None
    assert result[0] == TradeDirection.CALL
    assert entry["metrics"].get("gate_reason") is None
    assert entry["metrics"].get("calibration_mode") == "calibrated"


def test_quality_cluster_fail_branch_via_mock():
    with patch(
        "src.application.services.execution_quality_gate_cluster.passes_execution_quality",
        return_value=False,
    ):
        orch = SimpleNamespace(
            config={"orchestrator": {"execution": {"mandatory_trade_each_cycle": False}}},
            risk_manager=None,
            _quality_skipped_cycles_counter=0,
        )
        decisions = {"R_10": {"metrics": {"deploy_ok": True, "calibrated_prob": 0.55}}}
        assert quality_conviction_suspends_cluster(orch, decisions) is False


def test_hurst_regime_allowed_when_gating_disabled():
    assert hurst_regime_allowed(0.50, {"enabled": False}) is True


def test_candidate_block_reason_quality_paths():
    assert _candidate_block_reason({"quality_guard_reject": True}) == "quality_guard_reject"
    assert _candidate_block_reason({"signal_status": "SIGNAL_SUSPENDED"}) == "SIGNAL_SUSPENDED"


def test_propagate_cluster_signal_suspended_creates_metrics():
    decisions = {"R_10": {"direction": TradeDirection.CALL}}
    propagate_cluster_signal_suspended(decisions)
    assert decisions["R_10"]["metrics"]["signal_status"] == SIGNAL_SUSPENDED


def test_risk_session_bankroll_pending_from_map():
    rm = SimpleNamespace(initial_bankroll=100.0, pending_loss={"R_10": 12.5}, soft_recovery_config=None)
    bankroll, pending, soft = risk_session_bankroll_pending(rm)
    assert bankroll == 100.0
    assert pending == 12.5
    assert soft is None


def test_apply_quality_penalty_always_zero_when_unlocked():
    assert apply_quality_penalty_to_metrics({"calibrated_prob": 0.70}) == 0.0


def test_apply_meta_regression_edge_continues_when_veto_disabled():
    metrics = {"raw_prob": 0.40, "calibrated_prob": 0.55}
    direction, score = apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        0.20,
        meta_applied=True,
        base_score=0.55,
        symbol="R_10",
    )
    assert direction == TradeDirection.CALL
    assert score > 0.0


def test_collect_recovery_skip_waiver_empty_pool():
    orch = SimpleNamespace(
        anchor="R_10",
        symbols=["R_10", "R_50"],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False, "regime_evaluator": {"enabled": False}}},
            "risk_management": {"kelly": {}},
            "deep_learning": {},
        },
        risk_manager=SimpleNamespace(
            pending_loss={"R_10": 10.0},
            last_loss_symbol="R_10",
            consecutive_losses=2,
            consecutive_losses_linear=2,
            recovery_symbol_loss_streak={},
            symbol_loss_cooldown={"R_10": 1},
            proposal_skip_symbols=frozenset,
            pending_loss_total=lambda: 10.0,
            total_session_profit=0.0,
        ),
        _active_cycle_id=9,
        _recovery_skip_counter=0,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: ["R_10"],
    )
    with (
        patch(
            "src.application.services.orchestrator.execution_collect.gather_cluster_candidates",
            return_value=[],
        ),
        patch(
            "src.application.services.orchestrator.execution_collect.mandatory_fallback_if_empty",
            return_value=[],
        ),
        patch(
            "src.application.services.orchestrator.execution_collect.resolve_mandatory_ultimate_candidate",
            return_value=(None, []),
        ),
    ):
        orders = collect_cluster_orders(exec_mgr, {"R_10": {"direction": TradeDirection.PUT, "metrics": {}}})
    assert orders == []


def test_mandatory_fallback_quality_block_paths():
    exec_mgr = SimpleNamespace(
        orch=SimpleNamespace(config={"orchestrator": {"execution": {}}}),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: ["R_10"],
    )
    with patch(
        "src.application.services.orchestrator.execution_collect._quality_blocks_mandatory_fallback",
        return_value=True,
    ):
        assert mandatory_fallback_if_empty(exec_mgr, {}, []) == []
        assert resolve_mandatory_ultimate_candidate(exec_mgr, {}) == (None, None)


@pytest.mark.asyncio
async def test_trading_cycle_quality_skip_branch_via_mock(orch_ready):
    orch = orch_ready
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = False
    decisions = {"R_10": {"metrics": {"calibrated_prob": 0.55, "deploy_ok": True}}}
    with (
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=decisions,
        ),
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.quality_conviction_suspends_cluster",
            return_value=True,
        ),
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.record_quality_guard_cycle_skip"
        ) as mock_record,
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.await_quality_skip_yield",
            new_callable=AsyncMock,
        ) as mock_skip,
        patch("src.application.services.orchestrator.trading_cycle_entry.mark_bar_processed", new_callable=AsyncMock),
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.await_regime_freeze_yield",
            new_callable=AsyncMock,
        ),
    ):
        await run_trading_cycle_if_ready(orch)
    mock_record.assert_called_once_with(orch)
    mock_skip.assert_awaited_once()

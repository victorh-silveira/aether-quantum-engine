"""Cobertura do Micro Passivo Residual e liberacao de EXEC_EMPTY em micro-banca."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.meta_payoff_veto_gate import should_veto_meta_payoff_negative_zscore
from src.application.services.orchestrator.orchestrator_run_loop import (
    _recovery_pending_total,
    align_exec_empty_recovery_signature_cooldown,
)
from src.application.services.orchestrator.post_settlement_cycle import (
    _await_exec_empty_signature_alignment,
)
from src.domain.models.trade import TradeDirection
from src.domain.risk.risk_manager import RiskManager
from src.domain.risk.soft_recovery_policy import (
    cointegration_valve_suppressed,
    configured_max_safe_stake_cap,
    gbdt_waiver_skip_threshold_for_risk,
    is_low_intensity_recovery,
    is_micro_residual_liability,
    negative_zscore_veto_floor_for_risk,
    resolve_gbdt_waiver_skip_threshold,
    resolve_negative_zscore_veto_floor,
    risk_session_bankroll_pending,
    soft_recovery_enabled,
)


def _soft_cfg() -> dict:
    return {
        "enabled": True,
        "max_safe_stake_cap": 4.20,
        "amort_cycles_min": 2,
        "amort_cycles_max": 5,
        "coing_redirect_drawdown_threshold": 15.00,
        "micro_residual_bankroll_max": 250.0,
        "micro_residual_pending_max": 5.0,
        "micro_residual_pending_pct": 0.05,
        "micro_residual_zscore_floor": -0.60,
        "gbdt_waiver_skip_cycles": 30,
        "micro_residual_gbdt_waiver_skips": 6,
    }


def _risk_config() -> dict:
    return {
        "kelly": {
            "fraction": 0.005,
            "max_stake_pct": 0.035,
            "max_bankroll_stake_fraction": 0.035,
            "dynamic_win_rate": False,
            "consensus_penalty_enabled": False,
            "stop_win_kelly_enabled": False,
            "recovery_sizing_conviction": 0.50,
            "recovery_min_conviction": 0.50,
            "recovery_min_val_accuracy": 0.50,
        },
        "soft_recovery": _soft_cfg(),
        "params": {"payout_estimate": 0.95, "stake_min": 1.0},
        "small_account_threshold": 100.0,
        "small_account_stop_win": 10.0,
    }


def _micro_rm(*, pending: float = 3.59, bankroll: float = 100.0, linear: int = 2) -> RiskManager:
    rm = RiskManager(_risk_config())
    rm.initial_bankroll = float(bankroll)
    rm.dlambert_unit = 1.0
    rm.consecutive_losses_linear = int(linear)
    rm.pending_loss = {"R_10": float(pending)}
    rm.last_loss_stake = float(pending)
    rm.total_session_profit = -float(pending)
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.62)
    rm._recovery_allowed = MagicMock(return_value=True)
    return rm


def test_micro_residual_liability_thresholds() -> None:
    soft = _soft_cfg()
    assert is_micro_residual_liability(100.0, 3.59, soft_recovery=soft) is True
    assert is_low_intensity_recovery(100.0, 3.59, soft_recovery=soft) is True
    assert is_micro_residual_liability(100.0, 5.00, soft_recovery=soft) is False
    assert is_micro_residual_liability(100.0, 5.01, soft_recovery=soft) is False
    assert is_micro_residual_liability(300.0, 3.59, soft_recovery=soft) is False
    assert resolve_negative_zscore_veto_floor(100.0, 3.59, soft_recovery=soft) == pytest.approx(-0.60)
    assert resolve_negative_zscore_veto_floor(100.0, 12.0, soft_recovery=soft) == pytest.approx(-0.20)
    assert resolve_gbdt_waiver_skip_threshold(100.0, 3.59, soft_recovery=soft) == 6
    assert resolve_gbdt_waiver_skip_threshold(100.0, 12.0, soft_recovery=soft) == 30
    assert cointegration_valve_suppressed(100.0, 3.59, soft_recovery=soft) is True


def test_micro_residual_payoff_veto_tolerates_mild_negative_z() -> None:
    rm = _micro_rm()
    metrics = {
        "predicted_payoff_edge": 0.12,
        "edge_expectancy": "NO_EDGE_NEUTRAL",
        "meta_payoff_edge_zscore": -0.45,
        "edge_zscore": -0.45,
        "raw_prob": 0.72,
    }
    assert should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.CALL, risk_manager=rm) is False
    assert metrics["meta_payoff_veto_zscore_floor"] == pytest.approx(-0.60)


def test_micro_residual_recovery_releases_gate_and_sizes_stake() -> None:
    rm = _micro_rm(pending=3.59, bankroll=100.0)
    assert rm.cointegration_redirect_active() is False
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "deploy_ok": True,
            "execute": True,
            "raw_prob": 0.72,
            "calibrated_prob": 0.72,
            "predicted_payoff_edge": 0.14,
            "meta_classifier_applied": True,
            "edge_expectancy": "NO_EDGE_NEUTRAL",
            "meta_payoff_edge_zscore": -0.45,
            "edge_zscore": -0.45,
            "edge_zscore_samples": 8,
            "val_accuracy": 0.66,
            "trade_score": 0.70,
            "call_votes": 7,
            "put_votes": 1,
            "indicators": {"hurst": 0.62, "bb_width": 0.14, "atr_norm": 0.02},
        },
    }
    with (
        patch(
            "src.application.services.execution_direction_resolver.resolve_meta_payoff_edge",
            return_value=(0.14, True),
        ),
        patch(
            "src.application.services.execution_direction_checks.passes_execution_quality",
            return_value=True,
        ),
    ):
        result = resolve_execution_direction(
            entry,
            symbol="R_10",
            risk_manager=rm,
            recovery_active=True,
        )
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics.get("gate_reason") != "meta_payoff_negative_zscore_veto"
    assert metrics.get("signal_status") != "SKIP"
    stake = rm.calculate_stake(
        100.0,
        "R_10",
        0.72,
        silent=True,
        apply_stop_win=False,
        dl_metrics={
            "execute": True,
            "trade_score": 0.70,
            "val_accuracy": 0.66,
            "raw_prob": 0.72,
            "calibrated_prob": 0.72,
            "meta_payoff_edge_zscore": -0.45,
        },
    )
    assert stake >= 1.0
    assert stake <= 4.20


def test_align_exec_empty_recovery_signature_cooldown_sets_boundary() -> None:
    rm = _micro_rm()
    orch = MagicMock()
    orch._last_cycle_cluster_executed = False
    orch.risk_manager = rm
    orch.config = {"orchestrator": {"signature_boundary_seconds": 60}}
    orch._cooldown_until = 0.0
    orch._post_settlement_incomplete_streak = 2
    with patch(
        "src.application.services.orchestrator.orchestrator_run_loop.seconds_until_next_signature_boundary",
        return_value=42.5,
    ):
        delay = align_exec_empty_recovery_signature_cooldown(orch)
    assert delay == pytest.approx(42.5)
    assert orch._cooldown_until > 0.0
    assert orch._post_settlement_incomplete_streak == 0


def test_align_exec_empty_skips_when_cluster_executed_or_no_pending() -> None:
    orch = MagicMock()
    orch._last_cycle_cluster_executed = True
    orch.risk_manager = _micro_rm()
    assert align_exec_empty_recovery_signature_cooldown(orch) == 0.0
    orch._last_cycle_cluster_executed = False
    orch.risk_manager = _micro_rm(pending=0.0)
    orch.risk_manager.pending_loss = {}
    assert align_exec_empty_recovery_signature_cooldown(orch) == 0.0


def test_recovery_pending_total_from_pending_loss_map() -> None:
    orch = MagicMock()
    orch.risk_manager = None
    assert _recovery_pending_total(orch) == 0.0
    rm = SimpleNamespace(pending_loss={"R_10": 1.25, "R_50": 2.34})
    orch.risk_manager = rm
    assert _recovery_pending_total(orch) == pytest.approx(3.59)
    orch.risk_manager = SimpleNamespace(pending_loss="bad")
    assert _recovery_pending_total(orch) == 0.0


def test_soft_recovery_helpers_without_risk_manager() -> None:
    assert risk_session_bankroll_pending(None) == (0.0, 0.0, None)
    assert negative_zscore_veto_floor_for_risk(None) == pytest.approx(-0.20)
    assert gbdt_waiver_skip_threshold_for_risk(None) == 30
    assert soft_recovery_enabled({"soft_recovery": {"enabled": False}}) is False
    assert soft_recovery_enabled(soft_recovery={"enabled": True}) is True
    assert configured_max_safe_stake_cap({"max_safe_stake_cap": "x"}) is None
    assert configured_max_safe_stake_cap({"max_safe_stake_cap": -1.0}) is None
    assert configured_max_safe_stake_cap(None) is None


@pytest.mark.asyncio
async def test_post_settlement_signature_wait_on_recovery_pending() -> None:
    orch = MagicMock()
    orch.risk_manager = _micro_rm(pending=3.59)
    orch._cooldown_until = 0.0
    orch.running = True
    orch._post_settlement_wake = MagicMock()
    orch._post_settlement_wake.clear = MagicMock()
    orch._post_settlement_wake.wait = MagicMock()
    with (
        patch(
            "src.application.services.orchestrator.post_settlement_cycle.seconds_until_next_signature_boundary",
            return_value=0.05,
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._await_post_settlement_breath",
            new=AsyncMock(),
        ) as breath,
    ):
        await _await_exec_empty_signature_alignment(orch, 0.01)
        breath.assert_awaited()
        assert orch._cooldown_until > 0.0


@pytest.mark.asyncio
async def test_post_settlement_signature_wait_uses_pending_map_and_active_cooldown() -> None:
    orch = MagicMock()
    orch.risk_manager = SimpleNamespace(pending_loss={"R_10": 3.59})
    orch._cooldown_until = time.time() + 5.0
    orch.running = True
    with patch(
        "src.application.services.orchestrator.post_settlement_cycle._await_post_settlement_breath",
        new=AsyncMock(),
    ) as breath:
        await _await_exec_empty_signature_alignment(orch, 0.01)
        breath.assert_awaited()


@pytest.mark.asyncio
async def test_post_settlement_signature_wait_zero_delay_falls_back_to_poll() -> None:
    orch = MagicMock()
    orch.risk_manager = _micro_rm(pending=3.59)
    orch._cooldown_until = 0.0
    orch.running = True
    with (
        patch(
            "src.application.services.orchestrator.post_settlement_cycle.seconds_until_next_signature_boundary",
            return_value=0.0,
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._poll_delay",
            new=AsyncMock(),
        ) as poll,
    ):
        await _await_exec_empty_signature_alignment(orch, 0.01)
        poll.assert_awaited_once_with(0.01)

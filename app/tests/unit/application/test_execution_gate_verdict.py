"""Contrato gate_verdict e bloqueio de Single-Strike sob SOFT_SIZE."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.application.services.execution_gate_verdict import (
    VERDICT_ALLOW,
    VERDICT_HARD_SKIP,
    VERDICT_SOFT_SIZE,
    blocks_single_strike,
    is_soft_size,
    stamp_allow,
    stamp_hard_skip,
    stamp_soft_size,
)
from src.domain.risk.gate_verdict_sizing import blocks_single_strike_boost
from src.domain.risk.risk_stake_flow import apply_stop_win_kelly_boost


def test_stamp_hard_clears_neg_edge_soft():
    metrics = {"neg_edge_soft": True, "neg_edge_soft_kelly_mult": 0.55}
    stamp_hard_skip(metrics, "neg_edge")
    assert metrics["gate_verdict"] == VERDICT_HARD_SKIP
    assert metrics.get("neg_edge_soft") is None
    assert is_soft_size(metrics) is False


def test_stamp_soft_does_not_override_hard():
    metrics: dict = {}
    stamp_hard_skip(metrics, "anti_loss_rsi_momentum")
    stamp_soft_size(metrics, "neg_edge_soft")
    assert metrics["gate_verdict"] == VERDICT_HARD_SKIP


def test_stamp_allow_and_soft_flags_block_boost():
    metrics: dict = {}
    stamp_allow(metrics, "neg_edge_pass")
    assert metrics["gate_verdict"] == VERDICT_ALLOW
    assert blocks_single_strike(metrics) is False
    stamp_soft_size(metrics, "cal_margin")
    assert metrics["gate_verdict"] == VERDICT_SOFT_SIZE
    assert blocks_single_strike(metrics) is True
    assert blocks_single_strike_boost(metrics) is True
    assert blocks_single_strike_boost({"neg_edge_soft": True}) is True
    assert blocks_single_strike_boost(None) is False
    softed = {"gate_verdict": VERDICT_SOFT_SIZE}
    stamp_allow(softed, "ignored")
    assert softed["gate_verdict"] == VERDICT_SOFT_SIZE
    assert is_soft_size(None) is False
    assert is_soft_size({"loss_clf_soft": True}) is True


def test_apply_stop_win_kelly_boost_skips_under_soft_size():
    rm = MagicMock()
    rm.kelly_config = {
        "stop_win_kelly_min_conviction": 0.45,
        "stop_win_kelly_enabled": True,
        "stop_win_kelly_min_fraction": 1.0,
        "stop_win_kelly_max_fraction": 1.0,
        "stop_win_kelly_conviction_strong": 0.52,
        "stop_win_kelly_cycles_target": 1.0,
        "stop_win_kelly_live_n_min": 0,
        "soft_size_min_stake_pct": 0.025,
        "soft_size_max_stake_pct": 0.025,
        "soft_size_min_edge": 0.015,
    }
    rm.config = {
        "large_account_stop_win_pct": 4.31,
        "small_account_threshold": 50.0,
        "orchestrator": {"cycle_interval_seconds": 300},
        "compounding_rate_daily": 0.0431,
    }
    rm.initial_bankroll = 9700.0
    rm.total_session_profit = 0.0
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    soft_metrics = {
        "gate_verdict": "SOFT_SIZE",
        "neg_edge_soft": True,
        "neg_edge_tcn_cal_edge": 0.03,
        "live_n": 40,
        "live_wr": 0.55,
        "val_accuracy": 0.59,
    }
    kelly_base = 5.0
    out = apply_stop_win_kelly_boost(
        rm,
        kelly_base=kelly_base,
        bankroll=9700.0,
        payout=0.85,
        sizing_conviction=0.60,
        conviction=0.60,
        dl_execute=True,
        recovery_active=False,
        apply_stop_win=True,
        silent=True,
        live_metrics=soft_metrics,
    )
    assert out == pytest.approx(9700.0 * 0.025)
    assert soft_metrics.get("soft_size_stake_floor_applied") is True
    soft_pend = {
        "gate_verdict": "SOFT_SIZE",
        "anti_loss_soft": True,
        "neg_edge_tcn_cal_edge": 0.11,
        "live_n": 40,
        "live_wr": 0.55,
        "val_accuracy": 0.59,
    }
    out_pend = apply_stop_win_kelly_boost(
        rm,
        kelly_base=kelly_base,
        bankroll=9700.0,
        payout=0.85,
        sizing_conviction=0.60,
        conviction=0.60,
        dl_execute=True,
        recovery_active=True,
        apply_stop_win=True,
        silent=True,
        live_metrics=soft_pend,
    )
    assert out_pend == pytest.approx(9700.0 * 0.025)
    assert soft_pend.get("soft_size_stake_floor_applied") is True
    allow_metrics = {"gate_verdict": "ALLOW", "live_n": 40, "live_wr": 0.55, "val_accuracy": 0.59}
    boosted = apply_stop_win_kelly_boost(
        rm,
        kelly_base=kelly_base,
        bankroll=9700.0,
        payout=0.85,
        sizing_conviction=0.60,
        conviction=0.60,
        dl_execute=True,
        recovery_active=False,
        apply_stop_win=True,
        silent=True,
        live_metrics=allow_metrics,
    )
    assert boosted > kelly_base


def test_soft_size_stake_floor_respects_max_pct():
    from src.domain.risk.risk_stake_flow import apply_soft_size_stake_floor

    metrics = {"gate_verdict": "SOFT_SIZE", "neg_edge_tcn_cal_edge": 0.04}
    out = apply_soft_size_stake_floor(
        1.0,
        10000.0,
        {"soft_size_min_stake_pct": 0.04, "soft_size_max_stake_pct": 0.025, "soft_size_min_edge": 0.015},
        metrics,
    )
    assert out == pytest.approx(250.0)
    assert metrics.get("soft_size_min_stake_pct") == pytest.approx(0.025)


def test_soft_size_stake_floor_waives_on_edge_subfloor():
    from src.domain.risk.risk_stake_flow import apply_soft_size_stake_floor

    metrics = {"gate_verdict": "SOFT_SIZE", "neg_edge_tcn_cal_edge": 0.001}
    out = apply_soft_size_stake_floor(
        5.0,
        10000.0,
        {"soft_size_min_stake_pct": 0.025, "soft_size_max_stake_pct": 0.025, "soft_size_min_edge": 0.015},
        metrics,
    )
    assert out == pytest.approx(5.0)
    assert metrics.get("soft_size_stake_floor_waived") == "edge_subfloor"
    assert metrics.get("soft_size_stake_floor_applied") is None


def test_soft_size_stake_floor_waives_when_edge_missing():
    from src.domain.risk.risk_stake_flow import apply_soft_size_stake_floor

    metrics = {"gate_verdict": "SOFT_SIZE"}
    out = apply_soft_size_stake_floor(
        5.0,
        10000.0,
        {"soft_size_min_stake_pct": 0.025, "soft_size_max_stake_pct": 0.025, "soft_size_min_edge": 0.015},
        metrics,
    )
    assert out == pytest.approx(5.0)
    assert metrics.get("soft_size_stake_floor_waived") == "edge_subfloor"


def test_soft_size_cycle_edge_skips_invalid_and_non_dict():
    from src.domain.risk.risk_stake_flow import _soft_size_cycle_edge, apply_soft_size_stake_floor

    assert _soft_size_cycle_edge(None) is None
    assert _soft_size_cycle_edge("x") is None
    assert _soft_size_cycle_edge({"edge": "bad", "cal_edge": 0.02}) == pytest.approx(0.02)
    allow = {"gate_verdict": "ALLOW"}
    assert apply_soft_size_stake_floor(9.0, 1000.0, {"soft_size_min_stake_pct": 0.025}, allow) == 9.0
    soft = {"gate_verdict": "SOFT_SIZE", "neg_edge_tcn_cal_edge": 0.04}
    assert apply_soft_size_stake_floor(9.0, 1000.0, {"soft_size_min_stake_pct": 0.0}, soft) == 9.0
    assert apply_soft_size_stake_floor(9.0, 0.0, {"soft_size_min_stake_pct": 0.025}, soft) == 9.0


def test_apply_stop_win_kelly_boost_soft_early_return_without_boost_flags():
    rm = MagicMock()
    rm.kelly_config = {
        "stop_win_kelly_min_conviction": 0.45,
        "soft_size_min_stake_pct": 0.025,
        "soft_size_max_stake_pct": 0.025,
        "soft_size_min_edge": 0.015,
    }
    soft = {"gate_verdict": "SOFT_SIZE", "neg_edge_tcn_cal_edge": 0.03}
    out = apply_stop_win_kelly_boost(
        rm,
        kelly_base=7.0,
        bankroll=10000.0,
        payout=0.85,
        sizing_conviction=0.20,
        conviction=0.20,
        dl_execute=False,
        recovery_active=False,
        apply_stop_win=True,
        silent=True,
        live_metrics=soft,
    )
    assert out == pytest.approx(7.0)
    out_off = apply_stop_win_kelly_boost(
        rm,
        kelly_base=7.0,
        bankroll=10000.0,
        payout=0.85,
        sizing_conviction=0.60,
        conviction=0.60,
        dl_execute=True,
        recovery_active=False,
        apply_stop_win=False,
        silent=True,
        live_metrics=soft,
    )
    assert out_off == pytest.approx(7.0)

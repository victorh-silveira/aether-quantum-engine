"""Pass-through de direcao TCN com soft Kelly chop/edge e loss-clf."""

from unittest.mock import MagicMock

import pytest

from src.application.services.execution_direction_checks import (
    initial_direction_checks,
    is_technically_blocked,
)
from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.domain.models.trade import TradeDirection


def test_initial_direction_checks_allows_hurst_mid_band():
    entry = {
        "metrics": {
            "calibrated_prob": 0.62,
            "deploy_ok": True,
            "indicators": {"hurst": 0.50, "adx": 0.05},
        }
    }
    result = initial_direction_checks(entry, {})
    assert result is not None
    dl_dir, metrics, prob = result
    assert dl_dir == TradeDirection.CALL
    assert prob == 0.62
    assert metrics["resolved_direction"] == "CALL"


def test_is_technically_blocked_deploy_and_training():
    assert is_technically_blocked({"metrics": {"deploy_ok": False}}) is True
    assert is_technically_blocked({"metrics": {"gate_reason": "training"}}) is True
    assert is_technically_blocked({"metrics": {"deploy_ok": True, "calibrated_prob": 0.7}}) is False


def test_resolve_execution_direction_soft_negative_cal_edge():
    entry = {
        "metrics": {
            "calibrated_prob": 0.52,
            "deploy_ok": True,
            "predicted_payoff_edge": -0.20,
            "indicators": {"hurst": 0.60, "adx": 0.40},
            "kelly_fraction_scale": 1.0,
        }
    }
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.04},
        "risk_management": {"params": {"payout_estimate": 0.72}},
    }
    orch._log_dedupe = {}
    orch._active_cycle_id = 1
    result = resolve_execution_direction(entry, exec_cfg={}, symbol="R_10", orch=orch)
    assert result is not None
    _direction, metrics = result
    assert metrics.get("execution_candidate_ready") is True
    assert metrics.get("signal_status") != "SKIP:NEG_EDGE"
    assert metrics.get("neg_edge_soft") is True
    assert metrics["kelly_fraction_scale"] <= 0.55 + 1e-9


def test_resolve_execution_direction_soft_regime_chop():
    entry = {
        "metrics": {
            "calibrated_prob": 0.70,
            "deploy_ok": True,
            "indicators": {"hurst": 0.50, "adx": 0.10},
            "kelly_fraction_scale": 1.0,
        }
    }
    result = resolve_execution_direction(entry, exec_cfg={}, symbol="R_10")
    assert result is not None
    _direction, metrics = result
    assert metrics.get("execution_candidate_ready") is True
    assert metrics.get("signal_status") != "SKIP:REGIME_CHOP"
    assert metrics.get("regime_chop_soft") is True
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.55)


def test_resolve_execution_direction_blocks_technical():
    entry = {"metrics": {"calibrated_prob": 0.70, "deploy_ok": False, "gate_reason": "deploy"}}
    assert resolve_execution_direction(entry, exec_cfg={}) is None

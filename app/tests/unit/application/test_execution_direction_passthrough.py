"""Pass-through de direcao TCN com soft Kelly chop/edge e loss-clf."""

from unittest.mock import MagicMock, patch

import pytest

from src.application.services.execution_direction_checks import (
    initial_direction_checks,
    is_technically_blocked,
    seed_direction_metrics,
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


def test_seed_direction_metrics_preserves_raw_and_invalid_fallback():
    metrics = {"raw_prob": 0.37115, "calibrated_prob": 0.52}
    score = seed_direction_metrics(metrics, dl_dir=TradeDirection.CALL, prob=0.52)
    assert score == pytest.approx(0.52)
    assert metrics["raw_prob"] == pytest.approx(0.37115)
    assert metrics["calibrated_prob"] == pytest.approx(0.52)
    assert metrics["raw_call_prob"] == pytest.approx(0.37115)
    bad = {"raw_prob": object()}
    seed_direction_metrics(bad, dl_dir=TradeDirection.PUT, prob=0.45)
    assert bad["raw_prob"] == pytest.approx(0.45)
    assert bad["calibrated_prob"] == pytest.approx(0.45)


def test_resolve_execution_direction_soft_subfloor_cal_edge():
    entry = {
        "metrics": {
            "calibrated_prob": 0.59,
            "raw_prob": 0.59,
            "deploy_ok": True,
            "predicted_payoff_edge": 0.01,
            "indicators": {"hurst": 0.60, "adx": 0.40},
            "kelly_fraction_scale": 1.0,
            "loss_clf_auto_learn": True,
            "tcn_direction": "CALL",
        }
    }
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.04},
        "risk_management": {"params": {"payout_estimate": 0.72}},
        "orchestrator": {"execution": {"scale_vision": {"fusion_enabled": False}}},
    }
    orch._log_dedupe = {}
    orch._active_cycle_id = 1
    with patch(
        "src.application.services.execution_direction_resolver.apply_loss_classifier_gate",
        return_value=False,
    ):
        result = resolve_execution_direction(entry, exec_cfg={}, symbol="R_10", orch=orch)
    assert result is not None
    _direction, metrics = result
    assert metrics.get("execution_candidate_ready") is not False
    assert metrics.get("gate_reason") is None
    assert metrics.get("neg_edge_soft") is True
    assert float(metrics["raw_prob"]) == pytest.approx(0.59)
    assert float(metrics["calibrated_prob"]) == pytest.approx(0.59)
    assert "cal_side_edge" in metrics
    assert 0.0 < float(metrics["cal_side_edge"]) < 0.04


def test_resolve_execution_direction_hard_nonpositive_cal_edge():
    entry = {
        "metrics": {
            "calibrated_prob": 0.52,
            "raw_prob": 0.52,
            "deploy_ok": True,
            "predicted_payoff_edge": -0.20,
            "indicators": {"hurst": 0.60, "adx": 0.40},
            "kelly_fraction_scale": 1.0,
            "loss_clf_auto_learn": True,
            "tcn_direction": "CALL",
        }
    }
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.04},
        "risk_management": {"params": {"payout_estimate": 0.72}},
        "orchestrator": {"execution": {"scale_vision": {"fusion_enabled": False}}},
    }
    orch._log_dedupe = {}
    orch._active_cycle_id = 3
    with patch(
        "src.application.services.execution_direction_resolver.apply_loss_classifier_gate",
        return_value=False,
    ):
        result = resolve_execution_direction(entry, exec_cfg={}, symbol="R_10", orch=orch)
    assert result is not None
    _direction, metrics = result
    assert metrics.get("execution_candidate_ready") is False
    assert metrics.get("gate_reason") == "neg_edge"
    assert metrics.get("neg_edge_nonpositive_hard") is True
    assert metrics.get("neg_edge_soft") is None


def test_resolve_execution_direction_hard_neg_edge_override():
    entry = {
        "metrics": {
            "calibrated_prob": 0.52,
            "raw_prob": 0.37115,
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
        "orchestrator": {
            "execution": {
                "signal_skip": {
                    "neg_edge_hard_skip": True,
                    "neg_edge_soft_when_closed_candle_agree": False,
                    "neg_edge_soft_min_edge": -1.0,
                    "neg_edge_soft_kelly_mult": 0.55,
                },
                "scale_vision": {"fusion_enabled": False},
            }
        },
    }
    orch._log_dedupe = {}
    orch._active_cycle_id = 2
    result = resolve_execution_direction(entry, exec_cfg={}, symbol="R_10", orch=orch)
    assert result is not None
    _direction, metrics = result
    assert metrics.get("execution_candidate_ready") is False
    assert metrics.get("gate_reason") == "neg_edge"


def test_resolve_execution_direction_soft_regime_chop():
    entry = {
        "metrics": {
            "calibrated_prob": 0.70,
            "deploy_ok": True,
            "indicators": {"hurst": 0.50, "adx": 0.10},
            "kelly_fraction_scale": 1.0,
        }
    }
    orch = MagicMock()
    orch.config = {"orchestrator": {"execution": {"scale_vision": {"fusion_enabled": False}}}}
    orch._log_dedupe = {}
    orch._active_cycle_id = 3
    chop_cfg = {
        "chop_pause_enabled": True,
        "chop_adx_max": 0.22,
        "chop_hurst_min": 0.47,
        "chop_hurst_max": 0.53,
        "chop_soft_kelly_mult": 0.55,
    }
    with patch(
        "src.application.services.execution_regime_chop.parse_regime_chop_config",
        return_value=chop_cfg,
    ):
        result = resolve_execution_direction(entry, exec_cfg={}, symbol="R_10", orch=orch)
    assert result is not None
    _direction, metrics = result
    assert metrics.get("execution_candidate_ready") is True
    assert metrics.get("signal_status") != "SKIP:REGIME_CHOP"
    assert metrics.get("regime_chop_soft") is True
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.55)


def test_resolve_execution_direction_blocks_technical():
    entry = {"metrics": {"calibrated_prob": 0.70, "deploy_ok": False, "gate_reason": "deploy"}}
    assert resolve_execution_direction(entry, exec_cfg={}) is None

"""Pass-through de direcao TCN com SKIP tecnico e FLIP loss-clf."""

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


def test_resolve_execution_direction_passthrough_call():
    entry = {
        "metrics": {
            "calibrated_prob": 0.70,
            "raw_prob": 0.70,
            "deploy_ok": True,
            "predicted_payoff_edge": 0.06,
            "indicators": {"hurst": 0.60, "adx": 0.40},
            "kelly_fraction_scale": 1.0,
            "loss_clf_auto_learn": True,
            "tcn_direction": "CALL",
        }
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    orch._log_dedupe = {}
    orch._active_cycle_id = 1
    with patch(
        "src.application.services.execution_direction_resolver.apply_loss_classifier_gate",
        return_value=False,
    ):
        result = resolve_execution_direction(entry, exec_cfg={}, symbol="R_10", orch=orch)
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.CALL
    assert metrics.get("execution_candidate_ready") is True
    assert metrics.get("gate_reason") is None
    assert float(metrics["raw_prob"]) == pytest.approx(0.70)
    assert metrics["tcn_direction"] == "CALL"


def test_resolve_execution_direction_blocks_technical():
    entry = {"metrics": {"calibrated_prob": 0.70, "deploy_ok": False, "gate_reason": "deploy"}}
    assert resolve_execution_direction(entry, exec_cfg={}) is None


def test_resolve_execution_direction_skips_acc_floor():
    entry = {
        "metrics": {
            "calibrated_prob": 0.70,
            "raw_prob": 0.70,
            "val_accuracy": 0.50,
            "deploy_ok": True,
            "indicators": {"hurst": 0.60, "adx": 0.40},
        }
    }
    orch = MagicMock()
    orch.config = {"deep_learning": {"deploy_gate": {"soft_min_val_accuracy": 0.53}}}
    orch._log_dedupe = {}
    orch._active_cycle_id = 1
    result = resolve_execution_direction(
        entry,
        exec_cfg={"skip_below_soft_min_acc": True},
        symbol="R_10",
        orch=orch,
    )
    assert result is None
    assert entry["metrics"]["skip_reason"] == "acc_floor"


def test_resolve_execution_direction_inverts_exec_side_call_and_put():
    entry_call = {
        "metrics": {
            "calibrated_prob": 0.70,
            "raw_prob": 0.70,
            "deploy_ok": True,
            "tcn_direction": "CALL",
        }
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    orch._log_dedupe = {}
    with patch("src.application.services.execution_direction_resolver.apply_loss_classifier_gate", return_value=False):
        res_call = resolve_execution_direction(
            entry_call, exec_cfg={"invert_exec_side": True}, symbol="R_10", orch=orch
        )
    assert res_call is not None
    assert res_call[0] == TradeDirection.PUT
    assert res_call[1]["invert_exec_side"] is True

    entry_put = {
        "metrics": {
            "calibrated_prob": 0.30,
            "raw_prob": 0.30,
            "deploy_ok": True,
            "tcn_direction": "PUT",
        }
    }
    with patch("src.application.services.execution_direction_resolver.apply_loss_classifier_gate", return_value=False):
        res_put = resolve_execution_direction(entry_put, exec_cfg={"invert_exec_side": True}, symbol="R_10", orch=orch)
    assert res_put is not None
    assert res_put[0] == TradeDirection.CALL

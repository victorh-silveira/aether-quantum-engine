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


def test_resolve_execution_direction_flip_calculates_side_edge_from_p_eff():
    entry = {
        "metrics": {
            "calibrated_prob": 0.54,
            "raw_prob": 0.54,
            "deploy_ok": True,
            "loss_clf_flip": True,
            "loss_clf_p_eff": 0.60,
        }
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    orch._log_dedupe = {}
    with patch("src.application.services.execution_direction_resolver.apply_loss_classifier_gate", return_value=False):
        res = resolve_execution_direction(entry, exec_cfg={}, symbol="R_10", orch=orch)
    assert res is not None
    _dir, metrics = res
    assert metrics["cal_side_edge"] == pytest.approx(0.60 * 1.85 - 1.0)


def test_resolve_execution_direction_handles_invalid_loss_clf_p_eff_gracefully():
    entry = {
        "metrics": {
            "calibrated_prob": 0.60,
            "raw_prob": 0.60,
            "deploy_ok": True,
            "loss_clf_flip": True,
            "loss_clf_p_eff": "invalid",
        }
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    orch._log_dedupe = {}
    with patch("src.application.services.execution_direction_resolver.apply_loss_classifier_gate", return_value=False):
        res = resolve_execution_direction(entry, exec_cfg={}, symbol="R_10", orch=orch)
    assert res is not None


def test_resolve_execution_direction_anti_trend_lock_flips_direction():
    entry = {
        "metrics": {
            "calibrated_prob": 0.40,
            "raw_prob": 0.40,
            "deploy_ok": True,
            "pending_loss_total": 88.0,
        }
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    orch._log_dedupe = {}
    with (
        patch("src.application.services.execution_direction_resolver.apply_loss_classifier_gate", return_value=False),
        patch("src.application.services.execution_direction_resolver.should_anti_trend_lock_flip", return_value=True),
    ):
        res = resolve_execution_direction(entry, exec_cfg={}, symbol="1HZ75V", orch=orch)
    assert res is not None
    direction, metrics = res
    assert direction == TradeDirection.CALL
    assert metrics["anti_trend_lock_flip"] is True
    assert metrics["anti_trend_lock_from"] == "PUT"
    assert metrics["anti_trend_lock_to"] == "CALL"


def test_resolve_execution_direction_force():
    entry = {
        "metrics": {
            "calibrated_prob": 0.60,
            "deploy_ok": True,
            "signal_status": "WAITING",
        }
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    orch._log_dedupe = {}
    res = resolve_execution_direction(entry, exec_cfg={"force_trade_every_cycle": True}, symbol="1HZ75V", orch=orch)
    assert res is not None
    direction, metrics = res
    assert direction == TradeDirection.CALL
    assert metrics["force_trade_every_cycle"] is True
    assert "signal_status" not in metrics


def test_resolve_execution_direction_skips_counter_trend_without_edge():
    entry = {
        "metrics": {
            "calibrated_prob": 0.55,
            "deploy_ok": True,
            "trend_direction": "PUT",
            "closed_micro_candle_stamped": True,
            "closed_micro_candle_dir": "PUT",
            "cal_side_edge": 0.02,
        }
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    orch.risk_manager = None
    orch._log_dedupe = {}
    res = resolve_execution_direction(
        entry,
        exec_cfg={"skip_trend_discord": True, "skip_neg_edge": False},
        symbol="1HZ75V",
        orch=orch,
    )
    assert res is None
    assert entry["metrics"]["skip_reason"] == "counter_trend_unconfirmed"


def test_resolve_execution_direction_skips_on_exhaustion():
    entry = {
        "metrics": {
            "calibrated_prob": 0.55,
            "deploy_ok": True,
            "indicators": {"rsi": 0.82, "bb_pct_b": 1.15},
            "cal_side_edge": 0.02,
        }
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    orch.risk_manager = None
    orch._log_dedupe = {}
    res = resolve_execution_direction(
        entry,
        exec_cfg={"skip_exhaustion": True, "skip_neg_edge": False},
        symbol="1HZ75V",
        orch=orch,
    )
    assert res is None
    assert entry["metrics"]["skip_reason"] == "exhaustion_call"


def test_resolve_execution_direction_skips_on_chop_congestion():
    entry = {
        "metrics": {
            "calibrated_prob": 0.55,
            "deploy_ok": True,
            "indicators": {"adx": 0.12, "bb_width": 0.025},
            "cal_side_edge": 0.02,
        }
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    orch.risk_manager = None
    orch._log_dedupe = {}
    res = resolve_execution_direction(
        entry,
        exec_cfg={"skip_chop_congestion": True, "skip_neg_edge": False},
        symbol="1HZ75V",
        orch=orch,
    )
    assert res is None
    assert entry["metrics"]["skip_reason"] == "chop_congestion"


def test_resolve_execution_direction_blocks_chop_even_when_legacy_flag_is_true():
    entry = {
        "metrics": {
            "calibrated_prob": 0.55,
            "deploy_ok": True,
            "trend_direction": "CALL",
            "indicators": {"adx": 0.12, "bb_width": 0.025},
            "cal_side_edge": 0.02,
        }
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    orch.risk_manager = None
    orch._log_dedupe = {}
    res = resolve_execution_direction(
        entry,
        exec_cfg={"skip_chop_congestion": True, "skip_neg_edge": False},
        symbol="1HZ75V",
        orch=orch,
    )
    assert res is None
    assert entry["metrics"]["skip_reason"] == "chop_congestion"


def test_resolve_execution_direction_skips_on_two_bar_counter_trend():
    entry = {
        "metrics": {
            "calibrated_prob": 0.55,
            "deploy_ok": True,
            "scale_micro_prev_bar_dir": "PUT",
            "scale_micro_bar_dir": "PUT",
            "cal_side_edge": 0.02,
        }
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    orch.risk_manager = None
    orch._log_dedupe = {}
    res = resolve_execution_direction(
        entry,
        exec_cfg={"skip_two_bar_counter_trend": True, "skip_neg_edge": False},
        symbol="1HZ75V",
        orch=orch,
    )
    assert res is None
    assert entry["metrics"]["skip_reason"] == "two_bar_counter_trend"


def test_resolve_execution_direction_skips_on_wick_rejection():
    entry = {
        "metrics": {
            "calibrated_prob": 0.55,
            "deploy_ok": True,
            "closed_candle_ohlc": [100.0, 120.0, 95.0, 105.0],
            "cal_side_edge": 0.02,
        }
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    orch.risk_manager = None
    orch._log_dedupe = {}
    res = resolve_execution_direction(
        entry,
        exec_cfg={"skip_wick_rejection": True, "skip_neg_edge": False},
        symbol="1HZ75V",
        orch=orch,
    )
    assert res is None
    assert entry["metrics"]["skip_reason"] == "wick_rejection_call"


def test_resolve_execution_direction_skips_on_climactic_blowoff():
    entry = {
        "metrics": {
            "calibrated_prob": 0.55,
            "deploy_ok": True,
            "closed_candle_ohlc": [102.0, 140.0, 100.0, 138.0],
            "indicators": {"atr_abs": 10.0},
            "cal_side_edge": 0.02,
        }
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    orch.risk_manager = None
    orch._log_dedupe = {}
    res = resolve_execution_direction(
        entry,
        exec_cfg={"skip_climactic_blowoff": True, "skip_neg_edge": False},
        symbol="1HZ75V",
        orch=orch,
    )
    assert res is None
    assert entry["metrics"]["skip_reason"] == "climactic_blowoff_call"


def test_resolve_execution_direction_skips_on_adverse_tick_flow():
    entry = {
        "metrics": {
            "calibrated_prob": 0.55,
            "deploy_ok": True,
            "flow_features": {"price_velocity": -1.0, "micro_tick_acceleration": -0.8},
            "cal_side_edge": 0.02,
        }
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    orch.risk_manager = None
    orch._log_dedupe = {}
    res = resolve_execution_direction(
        entry,
        exec_cfg={"skip_adverse_tick_flow": True, "skip_neg_edge": False},
        symbol="1HZ75V",
        orch=orch,
    )
    assert res is None
    assert entry["metrics"]["skip_reason"] == "adverse_tick_flow_call"


def test_resolve_execution_direction_blocks_adverse_tick_flow():
    entry = {
        "metrics": {
            "calibrated_prob": 0.55,
            "deploy_ok": True,
            "flow_features": {"price_velocity": -1.0, "micro_tick_acceleration": -0.8},
            "cal_side_edge": 0.02,
        }
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    orch.risk_manager = None
    orch._log_dedupe = {}
    res = resolve_execution_direction(
        entry,
        exec_cfg={"skip_adverse_tick_flow": True, "skip_neg_edge": False},
        symbol="1HZ75V",
        orch=orch,
    )
    assert res is None
    metrics = entry["metrics"]
    assert metrics["skip_reason"] == "adverse_tick_flow_call"

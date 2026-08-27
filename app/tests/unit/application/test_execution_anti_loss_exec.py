"""Anti-loss live ancora em EXEC vs janela; replay dump 13:16."""

from __future__ import annotations

from src.application.services.execution_anti_loss import apply_anti_loss_seed_discord
from src.application.services.execution_neg_edge import apply_negative_cal_edge_pause
from src.application.services.execution_signal_skip import parse_signal_skip_config
from src.application.services.market_audit_log import format_gates_audit_line


def test_anti_loss_replay_c1_exec_put_window_put_passes():
    metrics = {
        "execution_candidate_ready": True,
        "tcn_direction": "CALL",
        "resolved_direction": "CALL",
        "exec_direction": "PUT",
        "calibrated_prob": 0.53726,
        "loss_clf_p_loss": 0.95783,
        "loss_clf_auto_learn": False,
        "ops_window_candle_dir": "PUT",
        "ops_window_candle_body": 3.911,
        "ops_window_stamped": True,
        "kelly_fraction_scale": 1.0,
    }
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    assert metrics.get("anti_loss_why") is None
    assert apply_negative_cal_edge_pause(metrics, min_edge=0.04, payout=0.72) is True
    assert metrics["execution_candidate_ready"] is False
    assert metrics.get("gate_reason") == "neg_edge"
    assert float(metrics["cal_side_edge"]) < 0.0


def test_anti_loss_replay_c6_confirm_weak_body():
    metrics = {
        "execution_candidate_ready": True,
        "tcn_direction": "CALL",
        "exec_direction": "CALL",
        "loss_clf_p_loss": 0.86117,
        "loss_clf_auto_learn": False,
        "ops_window_candle_dir": "CALL",
        "ops_window_candle_body": 0.035,
        "ops_window_stamped": True,
        "kelly_fraction_scale": 1.0,
    }
    cfg = parse_signal_skip_config({"anti_loss_live_confirm_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is True
    assert metrics["anti_loss_why"] == "live_confirm_weak"
    assert metrics.get("anti_loss_side") == "CALL"


def test_anti_loss_live_log_side_is_exec_not_tcn():
    metrics = {
        "execution_candidate_ready": True,
        "tcn_direction": "CALL",
        "exec_direction": "PUT",
        "ops_window_candle_dir": "CALL",
        "ops_window_candle_body": 0.80,
        "ops_window_stamped": True,
        "kelly_fraction_scale": 1.0,
    }
    cfg = parse_signal_skip_config({"anti_loss_live_exec_candle_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is True
    assert metrics["anti_loss_why"] == "live_exec_discord"
    assert metrics.get("anti_loss_side") == "PUT"
    assert metrics.get("anti_loss_tcn") == "CALL"
    line = format_gates_audit_line(metrics)
    assert "ANTI_LOSS skip why=live_exec_discord" in line
    assert "side=PUT" in line

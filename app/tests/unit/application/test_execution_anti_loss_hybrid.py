from unittest.mock import MagicMock

from src.application.services.execution_anti_loss import apply_anti_loss_seed_discord
from src.application.services.execution_signal_skip import parse_signal_skip_config
from tests.unit.application.test_execution_anti_loss import _base_metrics


def test_anti_loss_hybrid_anchor_stamps_telemetry():
    metrics = _base_metrics(
        ops_window_stamped=True,
        exec_direction="CALL",
        resolved_direction="CALL",
        ops_window_candle_dir="CALL",
        ops_window_candle_body=0.5,
        closed_micro_candle_dir="CALL",
        closed_micro_candle_body=0.8,
        indicators={"rsi": 0.50},
    )
    cfg = parse_signal_skip_config({"anti_loss_seed_discord_enabled": True})
    apply_anti_loss_seed_discord(metrics, cfg=cfg)
    assert metrics.get("anti_loss_anchor_mode") == "hybrid"
    assert metrics.get("anti_loss_anchor_agree") is True
    assert metrics.get("anti_loss_ops_dir") == "CALL"
    assert metrics.get("anti_loss_last_dir") == "CALL"


def test_anti_loss_seed_anchor_mode_is_ops_window():
    metrics = _base_metrics(
        ops_window_stamped=False,
        exec_direction="PUT",
        resolved_direction="PUT",
        tcn_direction="PUT",
        ops_window_candle_dir="CALL",
        ops_window_candle_body=0.2,
        closed_micro_candle_dir="CALL",
        closed_micro_candle_body=0.2,
        loss_clf_p_loss=0.92,
        loss_clf_auto_learn=False,
        fusion_blocked_tcn_pos_edge=True,
        indicators={"rsi": 0.50},
    )
    cfg = parse_signal_skip_config({"anti_loss_seed_discord_enabled": True, "anti_loss_hard_skip": True})
    apply_anti_loss_seed_discord(metrics, cfg=cfg)
    assert metrics.get("anti_loss_anchor_mode") == "ops_window"
    assert metrics.get("anti_loss_anchor_agree") is False


def test_anti_loss_hybrid_anchor_discord_reduces_body():
    metrics = _base_metrics(
        ops_window_stamped=True,
        exec_direction="CALL",
        resolved_direction="CALL",
        ops_window_candle_dir="CALL",
        ops_window_candle_body=0.5,
        closed_micro_candle_dir="PUT",
        closed_micro_candle_body=0.3,
        indicators={"rsi": 0.50},
    )
    cfg = parse_signal_skip_config(
        {"anti_loss_seed_discord_enabled": True, "anti_loss_live_confirm_enabled": True, "anti_loss_hard_skip": True}
    )
    apply_anti_loss_seed_discord(metrics, cfg=cfg)
    assert metrics.get("anti_loss_anchor_agree") is False


def test_anti_loss_ema_slope_soft_allows_exec(monkeypatch):
    from src.application.services import execution_anti_loss_live as live_mod

    monkeypatch.setattr(
        live_mod,
        "check_mini_ema_trend_and_slope",
        lambda *a, **k: (False, "anti_loss_ema_slope"),
    )
    metrics = _base_metrics(
        ops_window_stamped=True,
        exec_direction="CALL",
        resolved_direction="CALL",
        ops_window_candle_dir="CALL",
        ops_window_candle_body=0.5,
        indicators={"rsi": 0.50},
    )
    cfg = parse_signal_skip_config({"anti_loss_seed_discord_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, orch=MagicMock(), cfg=cfg) is False
    assert metrics.get("execution_candidate_ready") is not False
    assert metrics.get("anti_loss_soft") is True
    assert metrics.get("anti_loss_why") == "anti_loss_ema_slope"
    assert metrics.get("gate_verdict") == "SOFT_SIZE"
    assert metrics.get("gate_reason") is None

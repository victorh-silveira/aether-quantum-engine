"""Gate anti-loss live: soft Kelly em confirm/discord/weak/RSI; seed HARD."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.application.services.execution_anti_loss import (
    apply_anti_loss_seed_discord,
    evaluate_anti_loss_seed_discord,
)
from src.application.services.execution_signal_skip import parse_signal_skip_config


def _live_metrics(**extra):
    metrics = {
        "execution_candidate_ready": True,
        "tcn_direction": "PUT",
        "resolved_direction": "PUT",
        "exec_direction": "CALL",
        "loss_clf_p_loss": 0.34,
        "loss_clf_auto_learn": True,
        "closed_micro_candle_dir": "CALL",
        "fusion_blocked_tcn_pos_edge": True,
        "kelly_fraction_scale": 1.0,
        "pending_loss_total": 0.0,
        "indicators": {"rsi": 0.50},
    }
    metrics.update(extra)
    metrics.setdefault("ops_window_candle_dir", metrics.get("closed_micro_candle_dir"))
    metrics.setdefault("ops_window_candle_body", metrics.get("closed_micro_candle_body"))
    metrics.setdefault("ops_window_stamped", bool(metrics.get("closed_micro_candle_stamped")))
    return metrics


def _assert_soft(metrics, why: str) -> None:
    assert metrics.get("anti_loss_soft") is True
    assert metrics.get("anti_loss_why") == why
    assert metrics.get("gate_verdict") == "SOFT_SIZE"
    assert metrics.get("execution_candidate_ready") is not False
    assert metrics.get("gate_reason") is None


def test_anti_loss_live_strong_discord_hard_skip():
    metrics = _live_metrics(closed_micro_candle_body=0.863, closed_micro_candle_stamped=True)
    cfg = parse_signal_skip_config({})
    assert evaluate_anti_loss_seed_discord(metrics, cfg=cfg)["active"] is False
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False


def test_anti_loss_live_weak_discord_soft():
    metrics = _live_metrics(closed_micro_candle_body=0.015, closed_micro_candle_stamped=True)
    cfg = parse_signal_skip_config({"anti_loss_live_weak_candle_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    _assert_soft(metrics, "live_weak_candle")
    assert metrics.get("anti_loss_p_loss") is None


def test_anti_loss_live_weak_agree_soft():
    metrics = _live_metrics(
        closed_micro_candle_dir="PUT",
        closed_micro_candle_body=0.015,
        closed_micro_candle_stamped=True,
        exec_direction="PUT",
    )
    cfg = parse_signal_skip_config({"anti_loss_live_weak_candle_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    _assert_soft(metrics, "live_weak_candle")


def test_anti_loss_live_no_candle_soft():
    metrics = _live_metrics(closed_micro_candle_stamped=True)
    metrics.pop("closed_micro_candle_dir", None)
    metrics.pop("ops_window_candle_dir", None)
    cfg = parse_signal_skip_config({"anti_loss_live_weak_candle_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    _assert_soft(metrics, "live_no_candle")


def test_anti_loss_live_pos_edge_not_required_for_discord():
    metrics = _live_metrics(
        closed_micro_candle_body=0.12,
        closed_micro_candle_stamped=True,
        exec_direction="CALL",
        fusion_blocked_tcn_pos_edge=False,
        fusion_reason="ev_call",
        calibrated_prob=0.51,
        raw_prob=0.51,
    )
    cfg = parse_signal_skip_config({"anti_loss_live_confirm_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    _assert_soft(metrics, "live_confirm_weak")


def test_anti_loss_live_invalid_body_weak_soft():
    metrics = _live_metrics(closed_micro_candle_dir="CALL", closed_micro_candle_stamped=True)
    metrics.pop("closed_micro_candle_body", None)
    metrics.pop("ops_window_candle_body", None)
    cfg = parse_signal_skip_config({"anti_loss_live_weak_candle_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    _assert_soft(metrics, "live_weak_candle")


def test_anti_loss_live_unstamped_noop():
    metrics = _live_metrics(closed_micro_candle_body=0.038)
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False


def test_anti_loss_live_weak_disabled_noop():
    metrics = _live_metrics(closed_micro_candle_body=0.038)
    cfg = parse_signal_skip_config({"anti_loss_live_weak_candle_enabled": False})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False


def test_anti_loss_live_recover_pend_soft():
    metrics = _live_metrics(
        closed_micro_candle_body=0.015,
        closed_micro_candle_stamped=True,
        pending_loss_total=82.67,
    )
    cfg = parse_signal_skip_config({"anti_loss_live_weak_candle_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    _assert_soft(metrics, "live_weak_candle")


def test_anti_loss_live_discord_moderate_soft():
    metrics = _live_metrics(closed_micro_candle_body=0.12, closed_micro_candle_stamped=True)
    cfg = parse_signal_skip_config({"anti_loss_live_confirm_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    _assert_soft(metrics, "live_confirm_weak")


def test_anti_loss_live_discord_upper_band_soft():
    metrics = _live_metrics(closed_micro_candle_body=0.14, closed_micro_candle_stamped=True)
    cfg = parse_signal_skip_config({"anti_loss_live_confirm_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    _assert_soft(metrics, "live_confirm_weak")


def test_anti_loss_live_discord_c1_body_hard_skip():
    for body_val in (0.300, 0.360, 0.443, 0.314, 1.163):
        metrics = _live_metrics(closed_micro_candle_body=body_val, closed_micro_candle_stamped=True)
        assert apply_anti_loss_seed_discord(metrics, cfg=parse_signal_skip_config({})) is False


def test_anti_loss_live_confirm_c2_soft():
    metrics = _live_metrics(
        closed_micro_candle_dir="PUT",
        closed_micro_candle_body=0.12,
        closed_micro_candle_stamped=True,
        exec_direction="PUT",
    )
    cfg = parse_signal_skip_config({"anti_loss_live_confirm_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    _assert_soft(metrics, "live_confirm_weak")


def test_anti_loss_live_confirm_c5_soft():
    metrics = _live_metrics(
        closed_micro_candle_dir="PUT",
        closed_micro_candle_body=0.12,
        closed_micro_candle_stamped=True,
        exec_direction="PUT",
    )
    cfg = parse_signal_skip_config({"anti_loss_live_confirm_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    _assert_soft(metrics, "live_confirm_weak")


def test_anti_loss_live_confirm_c3_noop():
    metrics = _live_metrics(
        closed_micro_candle_dir="PUT",
        closed_micro_candle_body=0.750,
        closed_micro_candle_stamped=True,
        exec_direction="PUT",
    )
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False


def test_anti_loss_live_confirm_below_min_body_soft():
    metrics = _live_metrics(
        closed_micro_candle_dir="PUT",
        closed_micro_candle_body=0.12,
        closed_micro_candle_stamped=True,
        exec_direction="PUT",
    )
    cfg = parse_signal_skip_config({"anti_loss_live_confirm_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    _assert_soft(metrics, "live_confirm_weak")


def test_anti_loss_live_confirm_at_015_passes():
    metrics = _live_metrics(
        closed_micro_candle_dir="PUT",
        closed_micro_candle_body=0.15,
        closed_micro_candle_stamped=True,
        exec_direction="PUT",
        indicators={"rsi": 0.50},
    )
    cfg = parse_signal_skip_config({"anti_loss_live_confirm_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False


def test_anti_loss_live_exec_candle_discord_hard_skip():
    metrics = _live_metrics(
        closed_micro_candle_dir="PUT",
        closed_micro_candle_body=0.800,
        closed_micro_candle_stamped=True,
        exec_direction="CALL",
    )
    cfg = parse_signal_skip_config(
        {"anti_loss_live_exec_candle_enabled": True, "anti_loss_allow_candle_flip": False, "anti_loss_hard_skip": True}
    )
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is True
    assert metrics["anti_loss_why"] == "live_exec_discord"


def test_anti_loss_live_exec_candle_disabled_discord_soft():
    metrics = _live_metrics(
        closed_micro_candle_dir="PUT",
        closed_micro_candle_body=0.800,
        closed_micro_candle_stamped=True,
        exec_direction="CALL",
    )
    cfg = parse_signal_skip_config(
        {
            "anti_loss_live_exec_candle_enabled": False,
            "anti_loss_live_confirm_enabled": True,
            "anti_loss_hard_skip": True,
        }
    )
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    _assert_soft(metrics, "live_discord_weak")


def test_anti_loss_live_c70_fusion_ev_call_discord_soft():
    metrics = _live_metrics(
        closed_micro_candle_dir="CALL",
        closed_micro_candle_body=0.12,
        closed_micro_candle_stamped=True,
        exec_direction="CALL",
        fusion_blocked_tcn_pos_edge=False,
        fusion_reason="ev_call",
    )
    cfg = parse_signal_skip_config({"anti_loss_live_confirm_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    _assert_soft(metrics, "live_confirm_weak")


def test_anti_loss_live_invalid_exec_falls_through_confirm_soft():
    metrics = _live_metrics(
        closed_micro_candle_dir="PUT",
        closed_micro_candle_body=0.12,
        closed_micro_candle_stamped=True,
        exec_direction="SIDE",
    )
    cfg = parse_signal_skip_config({"anti_loss_live_confirm_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    _assert_soft(metrics, "live_confirm_weak")


def test_anti_loss_live_confirm_disabled_noop():
    metrics = _live_metrics(
        closed_micro_candle_dir="PUT",
        closed_micro_candle_body=0.269,
        closed_micro_candle_stamped=True,
        exec_direction="PUT",
    )
    cfg = parse_signal_skip_config({"anti_loss_live_confirm_enabled": False})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False


def test_anti_loss_seed_stamped_c2_confirm_weak_soft():
    metrics = _live_metrics(
        loss_clf_auto_learn=False,
        closed_micro_candle_dir="PUT",
        closed_micro_candle_body=0.12,
        closed_micro_candle_stamped=True,
        exec_direction="PUT",
        loss_clf_p_loss=0.87672,
    )
    cfg = parse_signal_skip_config({"anti_loss_live_confirm_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    _assert_soft(metrics, "live_confirm_weak")


def test_anti_loss_seed_unstamped_still_seed_discord():
    metrics = _live_metrics(
        loss_clf_auto_learn=False,
        closed_micro_candle_dir="CALL",
        closed_micro_candle_body=0.702,
        loss_clf_p_loss=0.88166,
    )
    assert apply_anti_loss_seed_discord(metrics, cfg=parse_signal_skip_config({"anti_loss_hard_skip": True})) is True


def test_anti_loss_ema_trend_soft_allows_exec():
    stream = MagicMock()
    closes = np.linspace(4800, 5000, 30)
    stream.get_mini_numpy_series.return_value = closes
    orch = MagicMock(stream=stream, symbols=["R_10"], anchor="R_10")
    metrics = _live_metrics(
        closed_micro_candle_stamped=True,
        exec_direction="PUT",
        resolved_direction="PUT",
        tcn_direction="PUT",
        closed_micro_candle_dir="CALL",
        ops_window_candle_dir="CALL",
        closed_micro_candle_body=2.5,
        indicators={"rsi": 0.50},
    )
    cfg = parse_signal_skip_config(
        {
            "anti_loss_hard_skip": True,
            "anti_loss_live_exec_candle_enabled": False,
            "anti_loss_allow_candle_flip": False,
        }
    )
    assert apply_anti_loss_seed_discord(metrics, orch=orch, cfg=cfg) is False
    _assert_soft(metrics, "anti_loss_ema_trend")


def test_anti_loss_live_ema_trend_flips_to_candle():
    stream = MagicMock()
    closes = np.linspace(4800, 5000, 30)
    stream.get_mini_numpy_series.return_value = closes
    orch = MagicMock(stream=stream, symbols=["R_10"], anchor="R_10")
    orch.config = {
        "deep_learning": {},
        "risk_management": {"params": {"payout_estimate": 0.85}},
        "orchestrator": {"execution": {"signal_skip": {"min_edge_explore": 0.015, "min_edge_recovery": 0.010}}},
    }
    metrics = _live_metrics(
        closed_micro_candle_stamped=True,
        exec_direction="PUT",
        resolved_direction="PUT",
        tcn_direction="PUT",
        closed_micro_candle_dir="CALL",
        ops_window_candle_dir="CALL",
        closed_micro_candle_body=2.5,
        calibrated_prob=0.62,
        indicators={"rsi": 0.50},
    )
    cfg = parse_signal_skip_config({"anti_loss_allow_candle_flip": True, "anti_loss_live_exec_candle_enabled": False})
    assert apply_anti_loss_seed_discord(metrics, orch=orch, cfg=cfg) is False
    assert metrics.get("anti_loss_flipped_to_candle") is True
    assert metrics.get("exec_direction") == "CALL"
    assert metrics.get("resolved_direction") == "CALL"
    assert metrics.get("anti_loss_why") == "live_exec_flip_to_candle"


def test_anti_loss_flip_syncs_fusion_p_eff_to_candle_side():
    stream = MagicMock()
    closes = np.linspace(4800, 5000, 30)
    stream.get_mini_numpy_series.return_value = closes
    orch = MagicMock(stream=stream, symbols=["R_10"], anchor="R_10")
    orch.config = {
        "deep_learning": {},
        "risk_management": {"params": {"payout_estimate": 0.85}},
        "orchestrator": {"execution": {"signal_skip": {"min_edge_explore": 0.015, "min_edge_recovery": 0.010}}},
    }
    metrics = _live_metrics(
        closed_micro_candle_stamped=True,
        exec_direction="PUT",
        resolved_direction="PUT",
        tcn_direction="PUT",
        closed_micro_candle_dir="CALL",
        ops_window_candle_dir="CALL",
        closed_micro_candle_body=2.5,
        calibrated_prob=0.62,
        indicators={"rsi": 0.50},
        fusion_applied=True,
        fusion_side="PUT",
        fusion_p_eff=0.70,
        fusion_p_call=0.58,
        fusion_p_put=0.70,
    )
    cfg = parse_signal_skip_config({"anti_loss_allow_candle_flip": True, "anti_loss_live_exec_candle_enabled": False})
    assert apply_anti_loss_seed_discord(metrics, orch=orch, cfg=cfg) is False
    assert metrics.get("anti_loss_flipped_to_candle") is True
    assert metrics.get("exec_direction") == "CALL"
    assert metrics.get("fusion_p_eff") == pytest.approx(0.58)
    assert metrics.get("fusion_side") == "PUT"


def test_anti_loss_flip_blocked_when_candle_edge_subfloor():
    stream = MagicMock()
    closes = np.linspace(4800, 5000, 30)
    stream.get_mini_numpy_series.return_value = closes
    orch = MagicMock(stream=stream, symbols=["R_10"], anchor="R_10")
    orch.config = {
        "deep_learning": {},
        "risk_management": {"params": {"payout_estimate": 0.85}},
        "orchestrator": {"execution": {"signal_skip": {"min_edge_explore": 0.015, "min_edge_recovery": 0.010}}},
    }
    metrics = _live_metrics(
        closed_micro_candle_stamped=True,
        exec_direction="PUT",
        resolved_direction="PUT",
        tcn_direction="PUT",
        closed_micro_candle_dir="CALL",
        ops_window_candle_dir="CALL",
        closed_micro_candle_body=2.5,
        calibrated_prob=0.545,
        indicators={"rsi": 0.50},
    )
    cfg = parse_signal_skip_config({"anti_loss_allow_candle_flip": True, "anti_loss_live_exec_candle_enabled": False})
    assert apply_anti_loss_seed_discord(metrics, orch=orch, cfg=cfg) is False
    assert metrics.get("anti_loss_flipped_to_candle") is not True
    assert metrics.get("exec_direction") == "PUT"
    assert metrics.get("anti_loss_flip_blocked") == "edge_subfloor"
    assert metrics.get("anti_loss_soft") is True
    assert metrics.get("anti_loss_why") == "anti_loss_ema_trend"


def test_anti_loss_flip_min_edge_fallback_explore_and_recovery():
    stream = MagicMock()
    closes = np.linspace(4800, 5000, 30)
    stream.get_mini_numpy_series.return_value = closes
    orch = MagicMock(stream=stream, symbols=["R_10"], anchor="R_10")
    orch.config = "not-a-dict"
    metrics = _live_metrics(
        closed_micro_candle_stamped=True,
        exec_direction="PUT",
        resolved_direction="PUT",
        tcn_direction="PUT",
        closed_micro_candle_dir="CALL",
        ops_window_candle_dir="CALL",
        closed_micro_candle_body=2.5,
        calibrated_prob=0.62,
        indicators={"rsi": 0.50},
    )
    cfg = parse_signal_skip_config({"anti_loss_allow_candle_flip": True, "anti_loss_live_exec_candle_enabled": False})
    assert apply_anti_loss_seed_discord(metrics, orch=orch, cfg=cfg) is False
    assert metrics.get("anti_loss_flipped_to_candle") is True
    assert metrics.get("anti_loss_flip_min_edge") == pytest.approx(0.015)
    metrics_rec = _live_metrics(
        closed_micro_candle_stamped=True,
        exec_direction="PUT",
        resolved_direction="PUT",
        tcn_direction="PUT",
        closed_micro_candle_dir="CALL",
        ops_window_candle_dir="CALL",
        closed_micro_candle_body=2.5,
        calibrated_prob=0.62,
        indicators={"rsi": 0.50},
        pending_loss_total=5.0,
    )
    assert apply_anti_loss_seed_discord(metrics_rec, orch=orch, cfg=cfg) is False
    assert metrics_rec.get("anti_loss_flipped_to_candle") is True
    assert metrics_rec.get("anti_loss_flip_min_edge") == pytest.approx(0.010)

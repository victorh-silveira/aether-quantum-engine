"""Testes unitarios para execution_market_confluence."""

from __future__ import annotations

from src.application.services.execution_market_confluence import (
    _extract_edge_float,
    _extract_indicator_float,
    should_skip_chop_congestion,
    should_skip_directional_momentum_discord,
    should_skip_exhaustion,
    should_skip_two_bar_momentum_trap,
)
from src.domain.models.trade import TradeDirection


def test_extract_indicator_float_from_indicators_dict():
    metrics = {"indicators": {"rsi": 0.65, "adx": "0.22"}}
    assert _extract_indicator_float(metrics, "rsi") == 0.65
    assert _extract_indicator_float(metrics, "adx") == 0.22


def test_extract_indicator_float_from_root():
    metrics = {"rsi": "0.72"}
    assert _extract_indicator_float(metrics, "rsi") == 0.72


def test_extract_indicator_float_missing_or_invalid():
    assert _extract_indicator_float({}, "rsi") is None
    assert _extract_indicator_float({"indicators": None}, "rsi") is None
    assert _extract_indicator_float({"indicators": "not_a_dict", "rsi": "bad"}, "rsi") is None


def test_extract_edge_float_fallbacks():
    assert _extract_edge_float({"cal_side_edge": 0.035}) == 0.035
    assert _extract_edge_float({"edge": "0.022"}) == 0.022
    assert _extract_edge_float({}) == 0.0
    assert _extract_edge_float({"cal_side_edge": "invalid"}) == 0.0


def test_should_skip_exhaustion_disabled_or_force():
    metrics = {"rsi": 0.80, "bb_pct_b": 1.10}
    cfg = {"skip_exhaustion": False}
    assert should_skip_exhaustion(metrics, TradeDirection.CALL, cfg) is False
    assert should_skip_exhaustion(metrics, TradeDirection.CALL, {"skip_exhaustion": True}, force=True) is False
    assert should_skip_exhaustion(metrics, TradeDirection.CALL, None) is False


def test_should_skip_exhaustion_protects_even_when_pend_or_flip():
    cfg = {"skip_exhaustion": True, "material_pending_min": 0.01}
    m1 = {"rsi": 0.80, "bb_pct_b": 1.10, "pending_loss_total": 50.0}
    assert should_skip_exhaustion(m1, TradeDirection.CALL, cfg) is True
    m2 = {"rsi": 0.80, "bb_pct_b": 1.10, "loss_clf_flip": True}
    assert should_skip_exhaustion(m2, TradeDirection.CALL, cfg) is True
    m3 = {"rsi": 0.80, "bb_pct_b": 1.10, "anti_trend_lock_flip": True}
    assert should_skip_exhaustion(m3, TradeDirection.CALL, cfg) is True


def test_should_skip_exhaustion_missing_indicators():
    cfg = {"skip_exhaustion": True}
    assert should_skip_exhaustion({"rsi": 0.80}, TradeDirection.CALL, cfg) is False
    assert should_skip_exhaustion({"bb_pct_b": 1.10}, TradeDirection.CALL, cfg) is False


def test_should_skip_exhaustion_call_overbought():
    cfg = {"skip_exhaustion": True}
    metrics = {"indicators": {"rsi": 0.78, "bb_pct_b": 1.08}, "edge": 0.02}
    assert should_skip_exhaustion(metrics, TradeDirection.CALL, cfg) is True
    assert metrics["skip_reason"] == "exhaustion_call"
    assert metrics["signal_status"] == "SKIP:exhaustion_call"
    assert metrics["execution_candidate_ready"] is False


def test_should_skip_exhaustion_call_super_edge_passes():
    cfg = {"skip_exhaustion": True}
    metrics = {"indicators": {"rsi": 0.78, "bb_pct_b": 1.08}, "cal_side_edge": 0.095}
    assert should_skip_exhaustion(metrics, TradeDirection.CALL, cfg) is False


def test_should_skip_exhaustion_put_oversold():
    cfg = {"skip_exhaustion": True}
    metrics = {"indicators": {"rsi": 0.20, "bb_pct_b": -0.10}, "edge": 0.01}
    assert should_skip_exhaustion(metrics, TradeDirection.PUT, cfg) is True
    assert metrics["skip_reason"] == "exhaustion_put"
    assert metrics["signal_status"] == "SKIP:exhaustion_put"


def test_should_skip_exhaustion_put_super_edge_passes():
    cfg = {"skip_exhaustion": True}
    metrics = {"indicators": {"rsi": 0.20, "bb_pct_b": -0.10}, "cal_side_edge": 0.085}
    assert should_skip_exhaustion(metrics, TradeDirection.PUT, cfg) is False


def test_should_skip_exhaustion_normal_levels():
    cfg = {"skip_exhaustion": True}
    metrics = {"indicators": {"rsi": 0.50, "bb_pct_b": 0.50}}
    assert should_skip_exhaustion(metrics, TradeDirection.CALL, cfg) is False
    assert should_skip_exhaustion(metrics, TradeDirection.PUT, cfg) is False


def test_should_skip_chop_congestion_disabled_or_force():
    metrics = {"adx": 0.10, "bb_width": 0.02}
    cfg = {"skip_chop_congestion": False}
    assert should_skip_chop_congestion(metrics, cfg) is False
    assert should_skip_chop_congestion(metrics, {"skip_chop_congestion": True}, force=True) is False
    assert should_skip_chop_congestion(metrics, None) is False


def test_should_skip_chop_congestion_protects_even_when_pend_or_flip():
    cfg = {"skip_chop_congestion": True, "material_pending_min": 0.01}
    m1 = {"adx": 0.10, "bb_width": 0.02, "pending_loss_total": 20.0}
    assert should_skip_chop_congestion(m1, cfg) is True
    m2 = {"adx": 0.10, "bb_width": 0.02, "loss_clf_flip": True}
    assert should_skip_chop_congestion(m2, cfg) is True
    m3 = {"adx": 0.10, "bb_width": 0.02, "anti_trend_lock_flip": True}
    assert should_skip_chop_congestion(m3, cfg) is True


def test_should_skip_chop_congestion_missing_indicators():
    cfg = {"skip_chop_congestion": True}
    assert should_skip_chop_congestion({"adx": None}, cfg) is False
    assert should_skip_chop_congestion({}, cfg) is False


def test_should_skip_chop_congestion_triggers():
    cfg = {"skip_chop_congestion": True}
    metrics = {"indicators": {"adx": 0.12, "bb_width_raw": 0.025}, "edge": 0.015}
    assert should_skip_chop_congestion(metrics, cfg) is True
    assert metrics["skip_reason"] == "chop_congestion"
    assert metrics["signal_status"] == "SKIP:chop_congestion"


def test_should_skip_chop_congestion_fallback_bb_width():
    cfg = {"skip_chop_congestion": True}
    metrics = {"indicators": {"adx": 0.12, "bb_width": 0.025}, "edge": 0.015}
    assert should_skip_chop_congestion(metrics, cfg) is True
    assert metrics["skip_reason"] == "chop_congestion"


def test_should_skip_chop_congestion_super_edge_passes():
    cfg = {"skip_chop_congestion": True}
    metrics = {"indicators": {"adx": 0.12, "bb_width": 0.025, "vol_ratio": 1.40}, "cal_side_edge": 0.22}
    assert should_skip_chop_congestion(metrics, cfg) is False


def test_should_skip_chop_congestion_squeeze_adx_below_22():
    cfg = {"skip_chop_congestion": True}
    metrics = {"indicators": {"adx": 0.21, "bb_width": -0.85}, "edge": 0.020}
    assert should_skip_chop_congestion(metrics, cfg) is True
    assert metrics["skip_reason"] == "chop_congestion"


def test_should_skip_chop_congestion_strong_trend_or_width():
    cfg = {"skip_chop_congestion": True}
    assert should_skip_chop_congestion({"adx": 0.25, "bb_width": 0.050}, cfg) is False
    assert should_skip_chop_congestion({"adx": 0.21, "bb_width": 0.080}, cfg) is False


def test_should_skip_two_bar_momentum_trap_disabled_or_force():
    metrics = {"scale_micro_prev_bar_dir": "PUT", "scale_micro_bar_dir": "PUT"}
    cfg = {"skip_two_bar_counter_trend": False}
    assert should_skip_two_bar_momentum_trap(metrics, TradeDirection.CALL, cfg) is False
    assert (
        should_skip_two_bar_momentum_trap(
            metrics, TradeDirection.CALL, {"skip_two_bar_counter_trend": True}, force=True
        )
        is False
    )
    assert should_skip_two_bar_momentum_trap(metrics, TradeDirection.CALL, None) is False


def test_should_skip_two_bar_momentum_trap_protects_even_when_pend_or_flip():
    cfg = {"skip_two_bar_counter_trend": True, "material_pending_min": 0.01}
    m1 = {"scale_micro_prev_bar_dir": "PUT", "scale_micro_bar_dir": "PUT", "pending_loss_total": 15.0}
    assert should_skip_two_bar_momentum_trap(m1, TradeDirection.CALL, cfg) is True
    m2 = {"scale_micro_prev_bar_dir": "PUT", "scale_micro_bar_dir": "PUT", "loss_clf_flip": True}
    assert should_skip_two_bar_momentum_trap(m2, TradeDirection.CALL, cfg) is True
    m3 = {"scale_micro_prev_bar_dir": "PUT", "scale_micro_bar_dir": "PUT", "anti_trend_lock_flip": True}
    assert should_skip_two_bar_momentum_trap(m3, TradeDirection.CALL, cfg) is True


def test_should_skip_two_bar_momentum_trap_invalid_bars():
    cfg = {"skip_two_bar_counter_trend": True}
    assert should_skip_two_bar_momentum_trap({}, TradeDirection.CALL, cfg) is False
    assert should_skip_two_bar_momentum_trap({"scale_micro_prev_bar_dir": "PUT"}, TradeDirection.CALL, cfg) is False
    assert (
        should_skip_two_bar_momentum_trap(
            {"scale_micro_prev_bar_dir": "INVALID", "scale_micro_bar_dir": "PUT"}, TradeDirection.CALL, cfg
        )
        is False
    )


def test_should_skip_two_bar_momentum_trap_triggers_call_against_puts():
    cfg = {"skip_two_bar_counter_trend": True}
    metrics = {"scale_micro_prev_bar_dir": "PUT", "scale_micro_bar_dir": "PUT", "edge": 0.03}
    assert should_skip_two_bar_momentum_trap(metrics, TradeDirection.CALL, cfg) is True
    assert metrics["skip_reason"] == "two_bar_counter_trend"
    assert metrics["signal_status"] == "SKIP:two_bar_counter_trend"


def test_should_skip_two_bar_momentum_trap_triggers_put_against_calls_with_candle_fallback():
    cfg = {"skip_two_bar_counter_trend": True}
    metrics = {
        "scale_micro_prev_bar_dir": "CALL",
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "CALL",
        "edge": 0.03,
    }
    assert should_skip_two_bar_momentum_trap(metrics, TradeDirection.PUT, cfg) is True
    assert metrics["skip_reason"] == "two_bar_counter_trend"


def test_should_skip_two_bar_momentum_trap_passes_aligned_or_mixed():
    cfg = {"skip_two_bar_counter_trend": True}
    m1 = {"scale_micro_prev_bar_dir": "CALL", "scale_micro_bar_dir": "CALL", "edge": 0.01}
    assert should_skip_two_bar_momentum_trap(m1, TradeDirection.CALL, cfg) is False
    m2 = {"scale_micro_prev_bar_dir": "PUT", "scale_micro_bar_dir": "CALL", "edge": 0.01}
    assert should_skip_two_bar_momentum_trap(m2, TradeDirection.CALL, cfg) is False


def test_should_skip_two_bar_momentum_trap_super_edge_passes():
    cfg = {"skip_two_bar_counter_trend": True}
    metrics = {"scale_micro_prev_bar_dir": "PUT", "scale_micro_bar_dir": "PUT", "cal_side_edge": 0.075}
    assert should_skip_two_bar_momentum_trap(metrics, TradeDirection.CALL, cfg) is False


def test_should_skip_directional_momentum_discord_disabled_or_force():
    metrics = {"di_diff": -0.20, "rsi": 0.40}
    cfg = {"skip_directional_momentum_discord": False}
    assert should_skip_directional_momentum_discord(metrics, TradeDirection.CALL, cfg) is False
    assert (
        should_skip_directional_momentum_discord(
            metrics, TradeDirection.CALL, {"skip_directional_momentum_discord": True}, force=True
        )
        is False
    )
    assert should_skip_directional_momentum_discord(metrics, TradeDirection.CALL, None) is False


def test_should_skip_directional_momentum_discord_missing_indicators():
    cfg = {"skip_directional_momentum_discord": True}
    assert should_skip_directional_momentum_discord({}, TradeDirection.CALL, cfg) is False
    assert should_skip_directional_momentum_discord({"di_diff": -0.20}, TradeDirection.CALL, cfg) is False
    assert should_skip_directional_momentum_discord({"rsi": 0.40}, TradeDirection.CALL, cfg) is False


def test_should_skip_directional_momentum_discord_super_edge_passes():
    cfg = {"skip_directional_momentum_discord": True}
    metrics = {"di_diff": -0.20, "rsi": 0.40, "cal_side_edge": 0.085}
    assert should_skip_directional_momentum_discord(metrics, TradeDirection.CALL, cfg) is False


def test_should_skip_directional_momentum_discord_call_triggers():
    cfg = {"skip_directional_momentum_discord": True}
    metrics = {"di_diff": -0.20, "rsi": 0.40, "edge": 0.03}
    assert should_skip_directional_momentum_discord(metrics, TradeDirection.CALL, cfg) is True
    assert metrics["skip_reason"] == "bearish_momentum_discord"
    assert metrics["signal_status"] == "SKIP:bearish_momentum_discord"


def test_should_skip_directional_momentum_discord_call_passes_if_not_violating():
    cfg = {"skip_directional_momentum_discord": True}
    # di_diff not severe enough
    assert (
        should_skip_directional_momentum_discord(
            {"di_diff": -0.10, "rsi": 0.40, "edge": 0.03}, TradeDirection.CALL, cfg
        )
        is False
    )
    # rsi not low enough
    assert (
        should_skip_directional_momentum_discord(
            {"di_diff": -0.20, "rsi": 0.50, "edge": 0.03}, TradeDirection.CALL, cfg
        )
        is False
    )


def test_should_skip_directional_momentum_discord_put_triggers():
    cfg = {"skip_directional_momentum_discord": True}
    metrics = {"di_diff": 0.20, "rsi": 0.60, "edge": 0.03}
    assert should_skip_directional_momentum_discord(metrics, TradeDirection.PUT, cfg) is True
    assert metrics["skip_reason"] == "bullish_momentum_discord"
    assert metrics["signal_status"] == "SKIP:bullish_momentum_discord"


def test_should_skip_directional_momentum_discord_put_passes_if_not_violating():
    cfg = {"skip_directional_momentum_discord": True}
    # di_diff not severe enough
    assert (
        should_skip_directional_momentum_discord({"di_diff": 0.10, "rsi": 0.60, "edge": 0.03}, TradeDirection.PUT, cfg)
        is False
    )
    # rsi not high enough
    assert (
        should_skip_directional_momentum_discord({"di_diff": 0.20, "rsi": 0.50, "edge": 0.03}, TradeDirection.PUT, cfg)
        is False
    )


def test_should_skip_two_bar_momentum_trap_pullback_with_trend_passes():
    cfg = {"skip_two_bar_counter_trend": True}
    metrics = {
        "scale_micro_prev_bar_dir": "PUT",
        "scale_micro_bar_dir": "PUT",
        "trend_direction": "CALL",
        "edge": 0.025,
    }
    assert should_skip_two_bar_momentum_trap(metrics, TradeDirection.CALL, cfg) is False


def test_should_skip_exhaustion_trend_confluence_passes():
    cfg = {"skip_exhaustion": True}
    metrics = {
        "indicators": {"rsi": 0.78, "bb_pct_b": 1.08},
        "edge": 0.025,
        "trend_direction": "CALL",
    }
    assert should_skip_exhaustion(metrics, TradeDirection.CALL, cfg) is False


def test_should_skip_chop_congestion_trend_confluence_passes():
    cfg = {"skip_chop_congestion": True}
    metrics = {
        "indicators": {"adx": 0.15, "bb_width": 0.02},
        "edge": 0.025,
        "trend_direction": "PUT",
        "exec_direction": "PUT",
    }
    assert should_skip_chop_congestion(metrics, cfg) is False


def test_should_skip_chop_congestion_edge_35_passes():
    cfg = {"skip_chop_congestion": True}
    metrics = {
        "indicators": {"adx": 0.15, "bb_width": 0.02},
        "edge": 0.038,
    }
    assert should_skip_chop_congestion(metrics, cfg) is False


def test_should_skip_directional_momentum_discord_trend_confluence_passes():
    cfg = {"skip_directional_momentum_discord": True}
    metrics = {
        "di_diff": -0.20,
        "rsi": 0.40,
        "edge": 0.025,
        "trend_direction": "CALL",
    }
    assert should_skip_directional_momentum_discord(metrics, TradeDirection.CALL, cfg) is False

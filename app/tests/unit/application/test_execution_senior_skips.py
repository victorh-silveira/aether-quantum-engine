"""Testes unitarios para execution_senior_skips."""

from __future__ import annotations

from src.application.services.execution_senior_skips import (
    apply_senior_execution_skips,
    resolve_senior_skip_decision,
)
from src.domain.models.trade import TradeDirection


def test_resolve_senior_skip_chop_congestion():
    # Trend alignment breakout
    res_trend = resolve_senior_skip_decision(TradeDirection.CALL, {"trend_direction": "PUT"}, "chop_congestion")
    assert res_trend == (TradeDirection.PUT, "chop_trend_breakout")

    # Oscillator rebound at support / resistance
    res_rsi_low = resolve_senior_skip_decision(TradeDirection.PUT, {"rsi": 0.35}, "chop_congestion")
    assert res_rsi_low == (TradeDirection.CALL, "chop_oscillator_bound")

    res_rsi_high = resolve_senior_skip_decision(TradeDirection.CALL, {"rsi": 0.65}, "chop_congestion")
    assert res_rsi_high == (TradeDirection.PUT, "chop_oscillator_bound")

    # Bollinger %B bounds
    res_bb_low = resolve_senior_skip_decision(TradeDirection.PUT, {"bb_pct_b": 0.20}, "chop_congestion")
    assert res_bb_low == (TradeDirection.CALL, "chop_bb_bound")

    res_bb_high = resolve_senior_skip_decision(TradeDirection.CALL, {"bb_pct_b": 0.80}, "chop_congestion")
    assert res_bb_high == (TradeDirection.PUT, "chop_bb_bound")

    # Fallback when no indicators available
    res_none = resolve_senior_skip_decision(TradeDirection.CALL, {}, "chop_congestion")
    assert res_none == (TradeDirection.CALL, "chop_maintained")


def test_resolve_senior_skip_exhaustion():
    assert resolve_senior_skip_decision(TradeDirection.CALL, {}, "exhaustion") == (
        TradeDirection.PUT,
        "exhaustion_reversal",
    )
    assert resolve_senior_skip_decision(TradeDirection.PUT, {}, "exhaustion") == (
        TradeDirection.CALL,
        "exhaustion_reversal",
    )


def test_resolve_senior_skip_trend_discord():
    res_trend = resolve_senior_skip_decision(TradeDirection.PUT, {"trend_direction": "CALL"}, "trend_discord")
    assert res_trend == (TradeDirection.CALL, "trend_discord_alignment")

    res_neut = resolve_senior_skip_decision(TradeDirection.PUT, {}, "trend_discord")
    assert res_neut == (TradeDirection.PUT, "trend_maintained")


def test_resolve_senior_skip_two_bar_momentum_trap():
    res_bar = resolve_senior_skip_decision(
        TradeDirection.CALL, {"scale_micro_prev_bar_dir": "PUT"}, "two_bar_momentum_trap"
    )
    assert res_bar == (TradeDirection.PUT, "two_bar_flow_alignment")

    res_inv = resolve_senior_skip_decision(TradeDirection.CALL, {}, "two_bar_momentum_trap")
    assert res_inv == (TradeDirection.PUT, "two_bar_inversion")


def test_resolve_senior_skip_directional_momentum_discord():
    res_bull = resolve_senior_skip_decision(TradeDirection.PUT, {"di_diff": 0.20}, "directional_momentum_discord")
    assert res_bull == (TradeDirection.CALL, "momentum_flow_alignment")

    res_bear = resolve_senior_skip_decision(TradeDirection.CALL, {"di_diff": -0.20}, "directional_momentum_discord")
    assert res_bear == (TradeDirection.PUT, "momentum_flow_alignment")

    res_none = resolve_senior_skip_decision(TradeDirection.CALL, {}, "directional_momentum_discord")
    assert res_none == (TradeDirection.PUT, "momentum_inversion")


def test_resolve_senior_skip_wick_rejection():
    # Upper wick larger -> PUT
    m_up = {"closed_candle_ohlc": [100.0, 120.0, 95.0, 105.0]}
    assert resolve_senior_skip_decision(TradeDirection.CALL, m_up, "wick_rejection") == (
        TradeDirection.PUT,
        "wick_rejection_reversal",
    )

    # Lower wick larger -> CALL
    m_dn = {"closed_candle_ohlc": [115.0, 120.0, 95.0, 114.0]}
    assert resolve_senior_skip_decision(TradeDirection.PUT, m_dn, "wick_rejection") == (
        TradeDirection.CALL,
        "wick_rejection_reversal",
    )

    # Flat range or missing ohlc -> inversion fallback
    m_flat = {"closed_candle_ohlc": [100.0, 100.0, 100.0, 100.0]}
    assert resolve_senior_skip_decision(TradeDirection.CALL, m_flat, "wick_rejection") == (
        TradeDirection.PUT,
        "wick_rejection_flip",
    )

    assert resolve_senior_skip_decision(TradeDirection.PUT, {}, "wick_rejection") == (
        TradeDirection.CALL,
        "wick_rejection_flip",
    )


def test_resolve_senior_skip_climactic_blowoff():
    m_bull = {"closed_candle_ohlc": [100.0, 130.0, 99.0, 128.0]}
    assert resolve_senior_skip_decision(TradeDirection.CALL, m_bull, "climactic_blowoff") == (
        TradeDirection.PUT,
        "climactic_mean_reversion",
    )

    m_bear = {"closed_candle_ohlc": [128.0, 130.0, 99.0, 100.0]}
    assert resolve_senior_skip_decision(TradeDirection.PUT, m_bear, "climactic_blowoff") == (
        TradeDirection.CALL,
        "climactic_mean_reversion",
    )

    assert resolve_senior_skip_decision(TradeDirection.CALL, {}, "climactic_blowoff") == (
        TradeDirection.PUT,
        "climactic_flip",
    )


def test_resolve_senior_skip_opposing_marubozu():
    m_bear = {"closed_candle_ohlc": [110.0, 110.0, 90.0, 91.0]}
    assert resolve_senior_skip_decision(TradeDirection.CALL, m_bear, "opposing_marubozu") == (
        TradeDirection.PUT,
        "marubozu_continuation",
    )

    m_bull = {"closed_candle_ohlc": [90.0, 110.0, 90.0, 109.0]}
    assert resolve_senior_skip_decision(TradeDirection.PUT, m_bull, "opposing_marubozu") == (
        TradeDirection.CALL,
        "marubozu_continuation",
    )

    assert resolve_senior_skip_decision(TradeDirection.CALL, {}, "opposing_marubozu") == (
        TradeDirection.PUT,
        "marubozu_flip",
    )


def test_resolve_senior_skip_adverse_tick_flow():
    m_pos = {"flow_features": {"price_velocity": 1.5}}
    assert resolve_senior_skip_decision(TradeDirection.PUT, m_pos, "adverse_tick_flow") == (
        TradeDirection.CALL,
        "tick_flow_alignment",
    )

    m_neg = {"flow_features": {"price_velocity": -1.5}}
    assert resolve_senior_skip_decision(TradeDirection.CALL, m_neg, "adverse_tick_flow") == (
        TradeDirection.PUT,
        "tick_flow_alignment",
    )

    m_bad = {"flow_features": {"price_velocity": "invalid"}}
    assert resolve_senior_skip_decision(TradeDirection.CALL, m_bad, "adverse_tick_flow") == (
        TradeDirection.CALL,
        "tick_flow_alignment",
    )


def test_resolve_senior_skip_unknown_fallback():
    assert resolve_senior_skip_decision(TradeDirection.CALL, {}, "unknown_skip") == (
        TradeDirection.CALL,
        "senior_edge_confluence",
    )


def test_apply_senior_execution_skips_no_skip():
    d, blocked = apply_senior_execution_skips(TradeDirection.CALL, {})
    assert d == TradeDirection.CALL
    assert blocked is False


def test_apply_senior_execution_skips_legacy_blocks():
    m = {"indicators": {"adx": 0.10, "bb_width": 0.02}}
    cfg = {"skip_chop_congestion": True, "senior_confluence_flip": False}
    d, blocked = apply_senior_execution_skips(TradeDirection.CALL, m, exec_cfg=cfg)
    assert d == TradeDirection.CALL
    assert blocked is True


def test_apply_senior_execution_skips_converts_active():
    m = {
        "indicators": {"adx": 0.10, "bb_width": 0.02},
        "trend_direction": "PUT",
    }
    cfg = {"skip_chop_congestion": True, "senior_confluence_flip": True}
    d, blocked = apply_senior_execution_skips(TradeDirection.CALL, m, exec_cfg=cfg)
    assert d == TradeDirection.PUT
    assert blocked is False
    assert m["senior_skip_converted"] == "chop_congestion"
    assert m["senior_confluence_reason"] == "chop_trend_breakout"
    assert m["senior_trader_flip"] is True
    assert m["senior_flip_from"] == "CALL"
    assert m["senior_flip_to"] == "PUT"
    assert m["direction_origin"] == "FLIP_SENIOR_CONFLUENCE"
    assert m["cal_side_edge"] > 0.0

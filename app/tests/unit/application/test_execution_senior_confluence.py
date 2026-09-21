"""Testes unitarios para execution_senior_confluence."""

from __future__ import annotations

from src.application.services.execution_senior_confluence import (
    _check_climactic_confluence,
    _check_marubozu_confluence,
    _check_momentum_confluence,
    _check_tick_flow_confluence,
    _check_wick_confluence,
    evaluate_senior_directional_decision,
)
from src.domain.models.trade import TradeDirection


def test_evaluate_senior_decision_skips_if_already_flipped():
    m1 = {"loss_clf_flip": True, "edge": 0.01}
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m1) == (TradeDirection.CALL, False, None)

    m2 = {"anti_trend_lock_flip": True, "edge": 0.01}
    assert evaluate_senior_directional_decision(TradeDirection.PUT, m2) == (TradeDirection.PUT, False, None)

    m3 = {"edge": 0.01, "trend_direction": "PUT", "closed_micro_candle_dir": "PUT", "closed_micro_candle_stamped": True}
    assert evaluate_senior_directional_decision(
        TradeDirection.CALL, m3, exec_cfg={"senior_confluence_flip": False}
    ) == (
        TradeDirection.CALL,
        False,
        None,
    )


def test_evaluate_senior_decision_respects_high_edge():
    m = {"edge": 0.040, "trend_direction": "CALL"}
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m) == (TradeDirection.CALL, False, None)


def test_check_marubozu_confluence():
    assert _check_marubozu_confluence(TradeDirection.CALL, (100.0, 100.0, 100.0, 100.0)) is None
    assert _check_marubozu_confluence(TradeDirection.CALL, (100.0, 110.0, 90.0, 102.0)) is None

    # Bearish marubozu against CALL (open=110, high=110, low=90, close=91) -> body=19/20=0.95, lower_wick=1/20=0.05
    res = _check_marubozu_confluence(TradeDirection.CALL, (110.0, 110.0, 90.0, 91.0))
    assert res == (TradeDirection.PUT, "opposing_marubozu")

    # Lower wick >= 0.15 passes
    assert _check_marubozu_confluence(TradeDirection.CALL, (110.0, 110.0, 90.0, 95.0)) is None

    # Bullish marubozu against CALL (aligned, not opposing)
    assert _check_marubozu_confluence(TradeDirection.CALL, (90.0, 110.0, 90.0, 109.0)) is None

    # Bullish marubozu against PUT (open=90, high=110, low=90, close=109) -> body=19/20=0.95, upper_wick=1/20=0.05
    res = _check_marubozu_confluence(TradeDirection.PUT, (90.0, 110.0, 90.0, 109.0))
    assert res == (TradeDirection.CALL, "opposing_marubozu")

    # Upper wick >= 0.15 passes
    assert _check_marubozu_confluence(TradeDirection.PUT, (90.0, 110.0, 90.0, 105.0)) is None

    # Bearish marubozu against PUT (aligned, not opposing)
    assert _check_marubozu_confluence(TradeDirection.PUT, (110.0, 110.0, 90.0, 91.0)) is None


def test_check_wick_confluence():
    assert _check_wick_confluence(TradeDirection.CALL, (100.0, 100.0, 100.0, 100.0)) is None

    # CALL with upper wick >= 0.45 (open=100, high=120, low=99, close=101) -> range=21, upper_wick=19 (0.90)
    res = _check_wick_confluence(TradeDirection.CALL, (100.0, 120.0, 99.0, 101.0))
    assert res == (TradeDirection.PUT, "wick_rejection")

    # Small wick passes
    assert _check_wick_confluence(TradeDirection.CALL, (100.0, 105.0, 95.0, 104.0)) is None

    # PUT with lower wick >= 0.45 (open=119, high=120, low=99, close=118) -> range=21, lower_wick=19 (0.90)
    res = _check_wick_confluence(TradeDirection.PUT, (119.0, 120.0, 99.0, 118.0))
    assert res == (TradeDirection.CALL, "wick_rejection")

    # Small lower wick passes
    assert _check_wick_confluence(TradeDirection.PUT, (105.0, 105.0, 95.0, 96.0)) is None


def test_check_climactic_confluence():
    ohlc = (100.0, 150.0, 99.0, 148.0)  # range=51
    # Missing/low atr
    assert _check_climactic_confluence(TradeDirection.CALL, ohlc, {}) is None
    assert _check_climactic_confluence(TradeDirection.CALL, ohlc, {"atr_norm": 0.2}) is None
    assert _check_climactic_confluence(TradeDirection.CALL, (100.0, 102.0, 99.0, 101.0), {"atr_norm": 10.0}) is None
    assert _check_climactic_confluence(TradeDirection.CALL, (100.0, 100.0, 100.0, 100.0), {"atr_norm": 10.0}) is None

    # CALL climactic exhaustion via RSI
    m_call_rsi = {"atr_norm": 10.0, "rsi": 0.80}
    assert _check_climactic_confluence(TradeDirection.CALL, ohlc, m_call_rsi) == (
        TradeDirection.PUT,
        "climactic_exhaustion",
    )

    # CALL climactic exhaustion via BB %b
    m_call_bb = {"atr_norm": 10.0, "bb_pct_b": 1.10}
    assert _check_climactic_confluence(TradeDirection.CALL, ohlc, m_call_bb) == (
        TradeDirection.PUT,
        "climactic_exhaustion",
    )

    # PUT climactic exhaustion
    ohlc_put = (148.0, 150.0, 99.0, 100.0)
    m_put_rsi = {"atr_norm": 10.0, "rsi": 0.20}
    assert _check_climactic_confluence(TradeDirection.PUT, ohlc_put, m_put_rsi) == (
        TradeDirection.CALL,
        "climactic_exhaustion",
    )

    m_put_bb = {"atr_norm": 10.0, "bb_pct_b": -0.10}
    assert _check_climactic_confluence(TradeDirection.PUT, ohlc_put, m_put_bb) == (
        TradeDirection.CALL,
        "climactic_exhaustion",
    )

    # Normal levels pass
    m_norm = {"atr_norm": 10.0, "rsi": 0.50, "bb_pct_b": 0.50}
    assert _check_climactic_confluence(TradeDirection.CALL, ohlc, m_norm) is None
    assert _check_climactic_confluence(TradeDirection.PUT, ohlc_put, m_norm) is None


def test_check_momentum_confluence():
    assert _check_momentum_confluence(TradeDirection.CALL, {}) is None

    m_bear = {"di_diff": -0.20, "rsi": 0.40}
    assert _check_momentum_confluence(TradeDirection.CALL, m_bear) == (
        TradeDirection.PUT,
        "directional_momentum",
    )

    m_bull = {"di_diff": 0.20, "rsi": 0.60}
    assert _check_momentum_confluence(TradeDirection.PUT, m_bull) == (
        TradeDirection.CALL,
        "directional_momentum",
    )

    m_neut = {"di_diff": 0.05, "rsi": 0.50}
    assert _check_momentum_confluence(TradeDirection.CALL, m_neut) is None
    assert _check_momentum_confluence(TradeDirection.PUT, m_neut) is None


def test_check_tick_flow_confluence():
    assert _check_tick_flow_confluence(TradeDirection.CALL, {}) is None
    assert _check_tick_flow_confluence(TradeDirection.CALL, {"flow_features": "bad"}) is None
    assert _check_tick_flow_confluence(TradeDirection.CALL, {"flow_features": {"price_velocity": "bad"}}) is None

    # CALL adverse tick flow (vel=-1.5, accel=0.0 -> score=-1.5 <= -1.2)
    m_call = {"flow_features": {"price_velocity": -1.5, "micro_tick_acceleration": 0.0}}
    assert _check_tick_flow_confluence(TradeDirection.CALL, m_call) == (
        TradeDirection.PUT,
        "adverse_tick_flow",
    )

    # PUT adverse tick flow (vel=1.5 -> score=1.5 >= 1.2)
    m_put = {"flow_features": {"price_velocity": 1.5, "micro_tick_acceleration": 0.0}}
    assert _check_tick_flow_confluence(TradeDirection.PUT, m_put) == (
        TradeDirection.CALL,
        "adverse_tick_flow",
    )

    # Mild score passes
    m_mild = {"flow_features": {"price_velocity": 0.5, "micro_tick_acceleration": 0.0}}
    assert _check_tick_flow_confluence(TradeDirection.CALL, m_mild) is None
    assert _check_tick_flow_confluence(TradeDirection.PUT, m_mild) is None


def test_evaluate_senior_decision_via_ohlc_branches():
    # 1. Marubozu branch
    m_maru = {"edge": 0.01, "closed_candle_ohlc": [110.0, 110.0, 90.0, 91.0]}
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m_maru) == (
        TradeDirection.PUT,
        True,
        "opposing_marubozu",
    )

    # 2. Wick branch
    m_wick = {"edge": 0.01, "closed_candle_ohlc": [100.0, 120.0, 99.0, 101.0]}
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m_wick) == (
        TradeDirection.PUT,
        True,
        "wick_rejection",
    )

    # 3. Climax branch
    m_climax = {
        "edge": 0.01,
        "closed_candle_ohlc": [100.0, 150.0, 99.0, 148.0],
        "atr_norm": 10.0,
        "rsi": 0.80,
    }
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m_climax) == (
        TradeDirection.PUT,
        True,
        "climactic_exhaustion",
    )


def test_evaluate_senior_decision_trend_candle_alignment():
    m = {
        "edge": 0.01,
        "trend_direction": "PUT",
        "closed_micro_candle_dir": "PUT",
        "closed_micro_candle_stamped": True,
    }
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m) == (
        TradeDirection.PUT,
        True,
        "trend_candle_alignment",
    )

    m2 = {
        "edge": 0.01,
        "trend_direction": "CALL",
        "closed_micro_candle_dir": "CALL",
        "closed_micro_candle_stamped": True,
    }
    assert evaluate_senior_directional_decision(TradeDirection.PUT, m2) == (
        TradeDirection.CALL,
        True,
        "trend_candle_alignment",
    )


def test_evaluate_senior_decision_momentum_branch():
    m = {"edge": 0.01, "di_diff": -0.20, "rsi": 0.40}
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m) == (
        TradeDirection.PUT,
        True,
        "directional_momentum",
    )


def test_evaluate_senior_decision_two_bar_counter_trend():
    m = {
        "edge": 0.01,
        "scale_micro_prev_bar_dir": "PUT",
        "scale_micro_bar_dir": "PUT",
    }
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m) == (
        TradeDirection.PUT,
        True,
        "two_bar_counter_trend",
    )

    m2 = {
        "edge": 0.01,
        "scale_micro_prev_bar_dir": "CALL",
        "scale_micro_bar_dir": "CALL",
    }
    assert evaluate_senior_directional_decision(TradeDirection.PUT, m2) == (
        TradeDirection.CALL,
        True,
        "two_bar_counter_trend",
    )


def test_evaluate_senior_decision_tick_flow_branch():
    m = {
        "edge": 0.01,
        "flow_features": {"price_velocity": -1.5, "micro_tick_acceleration": 0.0},
    }
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m) == (
        TradeDirection.PUT,
        True,
        "adverse_tick_flow",
    )


def test_evaluate_senior_decision_neutral_returns_original():
    m = {"edge": 0.01}
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m) == (TradeDirection.CALL, False, None)


def test_evaluate_senior_decision_anti_counter_trend_loss():
    from src.application.services.direction_loss_tracker import record_direction_outcome

    record_direction_outcome("1HZ75V", "PUT", won=False)
    m = {"trend_direction": "CALL", "edge": 0.15}
    res = evaluate_senior_directional_decision(TradeDirection.PUT, m, symbol="1HZ75V")
    assert res == (TradeDirection.CALL, True, "anti_counter_trend_loss")


def test_evaluate_senior_decision_loss_clf_macro_discord():
    m = {"trend_direction": "CALL", "loss_clf_p_loss": 0.58, "edge": 0.15}
    res = evaluate_senior_directional_decision(TradeDirection.PUT, m)
    assert res == (TradeDirection.CALL, True, "loss_clf_macro_discord")

    # Invalid p_loss format does not crash
    m_bad = {"trend_direction": "PUT", "loss_clf_p_loss": "invalid", "edge": 0.15}
    assert evaluate_senior_directional_decision(TradeDirection.PUT, m_bad) == (TradeDirection.PUT, False, None)


def test_evaluate_senior_decision_chop_confluence():
    from src.application.services.execution_senior_confluence import _check_chop_confluence

    # Not chop
    assert _check_chop_confluence(TradeDirection.PUT, {"adx": 0.35}) is None

    # Chop bounce at support (PUT inverted to CALL)
    m_chop_support = {"adx": 0.15, "rsi": 0.38}
    assert evaluate_senior_directional_decision(TradeDirection.PUT, m_chop_support) == (
        TradeDirection.CALL,
        True,
        "chop_support_bounce",
    )

    # Chop bounce via BB %b
    m_chop_bb = {"micro_chop_congestion": True, "bb_pct_b": 0.30}
    assert evaluate_senior_directional_decision(TradeDirection.PUT, m_chop_bb) == (
        TradeDirection.CALL,
        True,
        "chop_support_bounce",
    )

    # Chop reversal at resistance (CALL inverted to PUT)
    m_chop_res = {"adx": 0.15, "rsi": 0.62}
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m_chop_res) == (
        TradeDirection.PUT,
        True,
        "chop_resistance_reversal",
    )

    m_chop_res_bb = {"micro_chop_congestion": True, "bb_pct_b": 0.70}
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m_chop_res_bb) == (
        TradeDirection.PUT,
        True,
        "chop_resistance_reversal",
    )


def test_evaluate_senior_decision_trend_pullback_resumption():
    # Counter-trend CALL during PUT trend without reversal pattern -> trend_pullback_resumption
    m = {"trend_direction": "PUT", "edge": 0.040}
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m) == (
        TradeDirection.PUT,
        True,
        "trend_pullback_resumption",
    )

    # Counter-trend with invalid p_loss falls through to trend_pullback_resumption
    m_bad_ploss = {"trend_direction": "PUT", "loss_clf_p_loss": "bad_type"}
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m_bad_ploss) == (
        TradeDirection.PUT,
        True,
        "trend_pullback_resumption",
    )

    # Counter-trend with neutral ohlc (no climax, no wick) returns trend_pullback_resumption
    ohlc_neutral = [100.0, 102.0, 99.0, 101.0]
    m_neutral_ohlc = {"trend_direction": "PUT", "closed_candle_ohlc": ohlc_neutral}
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m_neutral_ohlc) == (
        TradeDirection.PUT,
        True,
        "trend_pullback_resumption",
    )

    # Counter-trend with genuine wick rejection passes to wick_rejection
    ohlc_wick = [117.0, 120.0, 99.0, 118.0]  # close > open (CALL), lower wick >= 0.45
    m_wick = {"trend_direction": "PUT", "closed_candle_ohlc": ohlc_wick}
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m_wick) == (
        TradeDirection.CALL,
        True,
        "wick_rejection",
    )

    # Genuine climactic exhaustion at extreme oversold allows mean-reversion
    ohlc_climax = [140.0, 150.0, 95.0, 105.0]
    m_climax = {
        "trend_direction": "PUT",
        "closed_candle_ohlc": ohlc_climax,
        "atr_norm": 10.0,
        "rsi": 0.15,
    }
    assert evaluate_senior_directional_decision(TradeDirection.CALL, m_climax) == (
        TradeDirection.CALL,
        True,
        "climactic_exhaustion",
    )


def test_resolve_closed_candle_direction():
    from src.application.services.execution_senior_confluence import resolve_closed_candle_direction

    # Direct from closed_micro_candle_dir
    assert resolve_closed_candle_direction({"closed_micro_candle_dir": "CALL"}) == "CALL"
    assert resolve_closed_candle_direction({"scale_micro_bar_dir": "PUT"}) == "PUT"

    # From OHLC
    assert resolve_closed_candle_direction({"closed_candle_ohlc": [100.0, 110.0, 95.0, 105.0]}) == "CALL"
    assert resolve_closed_candle_direction({"closed_candle_ohlc": [105.0, 110.0, 95.0, 100.0]}) == "PUT"

    # Flat or empty
    assert resolve_closed_candle_direction({"closed_candle_ohlc": [100.0, 100.0, 100.0, 100.0]}) is None
    assert resolve_closed_candle_direction({}) is None

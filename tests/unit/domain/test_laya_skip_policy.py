"""Testes da politica configuravel de SKIP para decisoes do Laya."""

from __future__ import annotations

from src.domain.indicators.technical_indicators import IndicatorSnapshot
from src.domain.models.laya_decision import LayaDecision
from src.domain.models.trade import TradeDirection
from src.domain.services.laya_skip_policy import evaluate_laya_skip


def _snapshot(*, candle_dir: float = 1.0, macd_hist_delta: float = 0.1, streak: float = 1.0) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        close=4900.0,
        ema_9=4900.0,
        ema_20=4900.0,
        ema_21=4900.0,
        ema_50=4899.0,
        ema_spread_atr=0.5,
        ema20_slope=0.1,
        ema50_slope=0.1,
        rsi_7=55.0,
        rsi_14=55.0,
        macd=0.5,
        macd_signal=0.3,
        macd_hist=0.2,
        macd_hist_delta=macd_hist_delta,
        adx_14=20.0,
        plus_di_14=22.0,
        minus_di_14=18.0,
        di_diff_14=4.0,
        adx_20=20.0,
        di_diff_20=4.0,
        cci_14=20.0,
        cmo_14=10.0,
        stoch_k=60.0,
        stoch_d=55.0,
        williams_r_14=-40.0,
        roc_14=0.05,
        bollinger_pct_b=0.6,
        bollinger_width=0.01,
        keltner_pct_b=0.6,
        atr_14=2.0,
        atr_50=2.0,
        atr_ratio=1.0,
        hurst_14=0.6,
        body_atr=0.5,
        upper_wick_atr=0.1,
        lower_wick_atr=0.1,
        candle_dir=candle_dir,
        streak=streak,
        session_zscore=0.5,
    )


def _decision(direction: TradeDirection = TradeDirection.CALL, prob: float = 0.52) -> LayaDecision:
    pc = prob if direction == TradeDirection.CALL else 1.0 - prob
    return LayaDecision("R_10", direction, prob, pc, 1.0 - pc, prob, 100.0, {})


def test_skip_policy_disabled_allows_trade():
    verdict = evaluate_laya_skip(
        _decision(prob=0.50), _snapshot(candle_dir=-1.0, macd_hist_delta=-0.2, streak=-5.0), {}
    )
    assert verdict.should_skip is False and verdict.reason == ""


def test_skip_policy_min_probability_blocks():
    cfg = {"laya": {"skip_policy": {"enabled": True, "min_probability": 0.51}}}
    verdict = evaluate_laya_skip(_decision(prob=0.505), _snapshot(), cfg)
    assert verdict.should_skip is True and "50.50%<51.00%" in verdict.reason


def test_skip_policy_min_body_atr_blocks_doji_and_allows_normal_candle():
    from dataclasses import replace

    cfg = {"laya": {"skip_policy": {"enabled": True, "min_body_atr": 0.15}}}
    doji = _snapshot()
    doji_low = replace(doji, body_atr=0.03)
    blocked = evaluate_laya_skip(_decision(prob=0.52), doji_low, cfg)
    assert blocked.should_skip is True and "vela_doji(body=0.03ATR<0.15)" in blocked.reason
    allowed = evaluate_laya_skip(_decision(prob=0.52), doji, cfg)
    assert allowed.should_skip is False


def test_skip_policy_counter_momentum_blocks_call_and_put():
    cfg = {"laya": {"skip_policy": {"enabled": True, "block_counter_momentum": True}}}
    call_verdict = evaluate_laya_skip(
        _decision(TradeDirection.CALL), _snapshot(candle_dir=-1.0, macd_hist_delta=-0.1), cfg
    )
    assert call_verdict.should_skip is True and call_verdict.reason == "contra_momentum_vela_e_macd"
    put_verdict = evaluate_laya_skip(_decision(TradeDirection.PUT), _snapshot(candle_dir=1.0, macd_hist_delta=0.1), cfg)
    assert put_verdict.should_skip is True and put_verdict.reason == "contra_momentum_vela_e_macd"
    put_hist_only = evaluate_laya_skip(
        _decision(TradeDirection.PUT), _snapshot(candle_dir=1.0, macd_hist_delta=0.0), cfg
    )
    assert put_hist_only.should_skip is True and put_hist_only.reason == "contra_momentum_vela_e_macd"


def test_skip_policy_bb_extreme_margin_blocks_chasing():
    from dataclasses import replace

    cfg = {"laya": {"skip_policy": {"enabled": True, "bb_extreme_margin": 0.05}}}
    snap_high = replace(_snapshot(), bollinger_pct_b=0.96)
    assert evaluate_laya_skip(_decision(TradeDirection.CALL), snap_high, cfg).reason == "extremo_banda(BB%B=0.96)"
    snap_low = replace(_snapshot(), bollinger_pct_b=0.01)
    assert evaluate_laya_skip(_decision(TradeDirection.PUT), snap_low, cfg).reason == "extremo_banda(BB%B=0.01)"
    snap_mid = replace(_snapshot(), bollinger_pct_b=0.50)
    assert evaluate_laya_skip(_decision(TradeDirection.CALL), snap_mid, cfg).should_skip is False


def test_skip_policy_delay_wick_and_volatility_guards():
    from dataclasses import replace

    cfg = {
        "laya": {
            "skip_policy": {
                "enabled": True,
                "max_entry_delay_seconds": 30,
                "max_counter_wick_atr": 0.75,
                "min_atr_ratio": 0.7,
                "max_atr_ratio": 1.8,
            }
        }
    }
    snap = _snapshot()
    delayed = evaluate_laya_skip(_decision(TradeDirection.CALL), snap, cfg, entry_delay_seconds=45.0)
    assert delayed.should_skip is True and delayed.reason == "atraso_vela(45s>30s)"

    call_wick = replace(snap, upper_wick_atr=0.85, body_atr=0.30)
    assert evaluate_laya_skip(_decision(TradeDirection.CALL), call_wick, cfg).reason == "pavio_rejeicao(0.85ATR>=0.75)"
    put_wick = replace(snap, lower_wick_atr=0.90, body_atr=0.25)
    assert evaluate_laya_skip(_decision(TradeDirection.PUT), put_wick, cfg).reason == "pavio_rejeicao(0.90ATR>=0.75)"

    low_vol = replace(snap, atr_ratio=0.55)
    assert evaluate_laya_skip(_decision(TradeDirection.CALL), low_vol, cfg).reason == "volatilidade_anomala(ratio=0.55)"
    high_vol = replace(snap, atr_ratio=2.10)
    assert evaluate_laya_skip(_decision(TradeDirection.PUT), high_vol, cfg).reason == "volatilidade_anomala(ratio=2.10)"

    ok = evaluate_laya_skip(_decision(TradeDirection.CALL), snap, cfg, entry_delay_seconds=10.0)
    assert ok.should_skip is False


def test_skip_policy_counter_streak_blocks_and_aligned_passes():
    cfg = {
        "laya": {
            "skip_policy": {
                "enabled": True,
                "min_probability": 0.5025,
                "block_counter_momentum": True,
                "max_counter_streak": 2,
            }
        }
    }
    blocked_call = evaluate_laya_skip(_decision(TradeDirection.CALL), _snapshot(streak=-2.0), cfg)
    assert blocked_call.should_skip is True and blocked_call.reason == "contra_streak(-2)"
    blocked_put = evaluate_laya_skip(
        _decision(TradeDirection.PUT), _snapshot(candle_dir=-1.0, macd_hist_delta=-0.1, streak=3.0), cfg
    )
    assert blocked_put.should_skip is True and blocked_put.reason == "contra_streak(+3)"
    allowed = evaluate_laya_skip(
        _decision(TradeDirection.CALL, prob=0.51), _snapshot(candle_dir=1.0, macd_hist_delta=0.1, streak=1.0), cfg
    )
    assert allowed.should_skip is False and allowed.reason == ""


def test_skip_policy_strong_counter_trend_requires_all_three():
    from dataclasses import replace

    cfg = {"laya": {"skip_policy": {"enabled": True, "block_strong_counter_trend": True}}}
    snap_neg = replace(_snapshot(candle_dir=-1.0, macd_hist_delta=-0.1, streak=-1.0), di_diff_14=-5.0)
    assert (
        evaluate_laya_skip(_decision(TradeDirection.CALL), snap_neg, cfg).reason == "contra_fluxo_triplo(vela+macd+di)"
    )
    snap_pos = replace(_snapshot(candle_dir=1.0, macd_hist_delta=0.1, streak=1.0), di_diff_14=5.0)
    assert (
        evaluate_laya_skip(_decision(TradeDirection.PUT), snap_pos, cfg).reason == "contra_fluxo_triplo(vela+macd+di)"
    )
    snap_two = replace(_snapshot(candle_dir=-1.0, macd_hist_delta=-0.1, streak=-1.0), di_diff_14=5.0)
    assert evaluate_laya_skip(_decision(TradeDirection.CALL), snap_two, cfg).should_skip is False

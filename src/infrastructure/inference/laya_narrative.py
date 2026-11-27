"""Traducao causal dos indicadores e velas M5 para narrativa tecnica Laya."""

from __future__ import annotations

from typing import Any


def build_technical_narrative(symbol: str, indicators: dict[str, Any], candles: list[Any]) -> str:
    """Converte indicadores M5 e ultimas velas em prosa tecnica concisa em ingles."""
    close = float(indicators.get("close", 0.0) or 0.0)
    ema9 = float(indicators.get("ema_9", close) or close)
    ema20 = float(indicators.get("ema_20", close) or close)
    ema50 = float(indicators.get("ema_50", close) or close)
    spread = float(indicators.get("ema_spread_atr", 0.0) or 0.0)
    sl20 = float(indicators.get("ema20_slope", 0.0) or 0.0)
    adx14 = float(indicators.get("adx_14", 0.0) or 0.0)
    di_diff = float(indicators.get("di_diff_14", 0.0) or 0.0)
    hurst = float(indicators.get("hurst_14", 0.5) or 0.5)
    hist = float(indicators.get("macd_hist", 0.0) or 0.0)
    dhist = float(indicators.get("macd_hist_delta", 0.0) or 0.0)
    rsi7 = float(indicators.get("rsi_7", 50.0) or 50.0)
    rsi14 = float(indicators.get("rsi_14", 50.0) or 50.0)
    stoch = float(indicators.get("stoch_k", 50.0) or 50.0)
    cci = float(indicators.get("cci_14", 0.0) or 0.0)
    bb_pct = float(indicators.get("bollinger_pct_b", 0.5) or 0.5)
    atr14 = float(indicators.get("atr_14", 0.0) or 0.0)
    atr_ratio = float(indicators.get("atr_ratio", 1.0) or 1.0)
    streak = float(indicators.get("streak", 0.0) or 0.0)
    body_atr = float(indicators.get("body_atr", 0.0) or 0.0)
    up_wick = float(indicators.get("upper_wick_atr", 0.0) or 0.0)
    lo_wick = float(indicators.get("lower_wick_atr", 0.0) or 0.0)
    zscore = float(indicators.get("session_zscore", 0.0) or 0.0)

    if close > ema20 > ema50 and sl20 >= 0.0:
        trend = "bullish"
    elif close < ema20 < ema50 and sl20 <= 0.0:
        trend = "bearish"
    else:
        trend = "mixed/transition"
    price_pos = (
        "above EMA9/EMA20"
        if close > ema9 and close > ema20
        else ("below EMA9/EMA20" if close < ema9 and close < ema20 else "between EMA9/EMA20")
    )
    slope = "rising" if sl20 > 0.0 else "falling"
    adx_s = "strong trend" if adx14 >= 25.0 else ("moderate trend" if adx14 >= 20.0 else "weak ranging")
    di_s = "buyers lead" if di_diff > 0.0 else "sellers lead"
    macd_s = (
        "bullish rising"
        if hist > 0.0 and dhist > 0.0
        else ("bullish fading" if hist > 0.0 else ("bearish falling" if dhist < 0.0 else "bearish fading"))
    )
    rsi_s = "overbought" if rsi14 >= 65.0 else ("oversold" if rsi14 <= 35.0 else "neutral")
    bb_s = "near upper band" if bb_pct >= 0.8 else ("near lower band" if bb_pct <= 0.2 else "mid-band")
    hurst_s = "trending" if hurst >= 0.55 else ("mean-reverting" if hurst <= 0.45 else "random")
    vol_s = "expanding" if atr_ratio >= 1.15 else ("compressed" if atr_ratio <= 0.85 else "normal")
    seq = ", ".join(f"{'UP' if bar.close >= bar.open else 'DOWN'}({bar.close - bar.open:+.2f})" for bar in candles[-6:])
    return (
        f"{symbol} M5 close={close:.2f}. "
        f"Trend: {trend} ({slope}, price {price_pos}), EMA9={ema9:.2f} EMA20={ema20:.2f} EMA50={ema50:.2f}, spread={spread:+.2f}ATR. "
        f"Regime: {adx_s} (ADX14={adx14:.1f}), {di_s} (dDI={di_diff:+.1f}), Hurst={hurst:.2f} ({hurst_s}). "
        f"Momentum: MACD hist={hist:+.2f} ({macd_s}, dHist={dhist:+.2f}). "
        f"Oscillators: RSI7={rsi7:.1f} RSI14={rsi14:.1f} ({rsi_s}), Stoch={stoch:.1f}, CCI={cci:.1f}, "
        f"BB%B={bb_pct:.2f} ({bb_s}), ATR={atr14:.2f} (ratio={atr_ratio:.2f} {vol_s}). "
        f"Last 6 candles: {seq}. Streak={streak:+.0f}, body={body_atr:.2f}ATR, "
        f"wicks={up_wick:.2f}/{lo_wick:.2f}ATR, Z={zscore:+.2f}."
    )

"""Decisao organica e adaptativa de direcao (fluxo vs exaustao/fade)."""

from __future__ import annotations

import math
from typing import Any


def _calc_ema(values: list[float], span: int) -> list[float]:
    """Calcula media movel exponencial para a serie."""
    alpha = 2.0 / (span + 1.0)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1.0 - alpha) * result[-1])
    return result


def _calc_atr(candles: list[Any]) -> float:
    """Calcula o Average True Range de 14 periodos."""
    if len(candles) < 15:
        return 1.0
    ranges = []
    for prev, curr in zip(candles[-15:-1], candles[-14:], strict=False):
        c_h, c_l, p_c = float(curr.high), float(curr.low), float(prev.close)
        ranges.append(max(c_h - c_l, abs(c_h - p_c), abs(c_l - p_c)))
    return sum(ranges) / 14.0 if ranges else 1.0


def _calc_rsi(closes: list[float], period: int = 14) -> float:
    """Calcula o Relative Strength Index (RSI)."""
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for prev, curr in zip(closes[-period - 1 : -1], closes[-period:], strict=False):
        chg = curr - prev
        gains.append(max(0.0, chg))
        losses.append(max(0.0, -chg))
    avg_g = sum(gains) / period
    avg_l = sum(losses) / period
    if avg_l == 0:
        return 100.0 if avg_g > 0 else 50.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))


def _calc_bollinger_pct_b(closes: list[float], period: int = 14, mult: float = 2.0) -> float:
    """Calcula a posicao percentual %B dentro das Bandas de Bollinger."""
    if len(closes) < period:
        return 0.5
    recent = closes[-period:]
    mean = sum(recent) / period
    variance = sum((x - mean) ** 2 for x in recent) / period
    std = math.sqrt(variance)
    upper = mean + mult * std
    lower = mean - mult * std
    width = upper - lower
    return (recent[-1] - lower) / width if width > 1e-10 else 0.5


def _calc_stochastic(candles: list[Any], period: int = 14, smooth: int = 3) -> tuple[float, float]:
    """Calcula os osciladores %K e %D do Estocastico."""
    if len(candles) < period + smooth:
        return 50.0, 50.0
    k_vals = []
    for i in range(smooth):
        sub = candles[-period - smooth + 1 + i : len(candles) - smooth + 1 + i]
        hh = max(float(b.high) for b in sub)
        ll = min(float(b.low) for b in sub)
        c = float(sub[-1].close)
        k_vals.append(((c - ll) / (hh - ll) * 100.0) if hh > ll else 50.0)
    pct_k = k_vals[-1]
    pct_d = sum(k_vals) / len(k_vals) if k_vals else 50.0
    return pct_k, pct_d


def _calc_macd(
    closes: list[float], fast_span: int = 7, slow_span: int = 14, signal_span: int = 5
) -> tuple[float, float, float]:
    """Calcula a linha MACD, sinal e histograma."""
    if len(closes) < slow_span + signal_span:
        return 0.0, 0.0, 0.0
    fast, slow = _calc_ema(closes, fast_span), _calc_ema(closes, slow_span)
    macd_line = [f - s for f, s in zip(fast, slow, strict=True)]
    sig_line = _calc_ema(macd_line, signal_span)
    return macd_line[-1], sig_line[-1], macd_line[-1] - sig_line[-1]


def _calc_cci(candles: list[Any], period: int = 14) -> float:
    """Calcula o Commodity Channel Index (CCI)."""
    if len(candles) < period:
        return 0.0
    recent = candles[-period:]
    tp_list = [(float(b.high) + float(b.low) + float(b.close)) / 3.0 for b in recent]
    mean_tp = sum(tp_list) / period
    mean_dev = sum(abs(tp - mean_tp) for tp in tp_list) / period
    return ((tp_list[-1] - mean_tp) / (0.015 * mean_dev)) if mean_dev > 1e-10 else 0.0


def _calc_adx(candles: list[Any], period: int = 14) -> float:
    """Calcula o Average Directional Index (ADX)."""
    if len(candles) < period + 1:
        return 20.0
    tr_s, pdm_s, mdm_s = 0.0, 0.0, 0.0
    for prev, curr in zip(candles[-period - 1 : -1], candles[-period:], strict=False):
        tr = max(
            float(curr.high) - float(curr.low),
            abs(float(curr.high) - float(prev.close)),
            abs(float(curr.low) - float(prev.close)),
        )
        tr_s += tr
        up = float(curr.high) - float(prev.high)
        down = float(prev.low) - float(curr.low)
        if up > down and up > 0:
            pdm_s += up
        if down > up and down > 0:
            mdm_s += down
    if tr_s <= 0:
        return 0.0
    pdi = (pdm_s / tr_s) * 100.0
    mdi = (mdm_s / tr_s) * 100.0
    denom = pdi + mdi
    return (abs(pdi - mdi) / denom * 100.0) if denom > 0 else 0.0


def resolve_organic_direction(
    candles: list[Any],
    signal_side: str,
) -> tuple[str, str]:
    """Decide organicamente a direcao real do contrato alinhada a tendencia de pullback institucional."""
    if not candles or len(candles) < 30:
        return signal_side, "fallback_poucas_velas"

    closes = [float(c.close) for c in candles]
    cf = candles[-1]
    prev_c = candles[-2]
    cf_open, cf_close = float(cf.open), float(cf.close)
    cf_high, cf_low = float(cf.high), float(cf.low)
    body = abs(cf_close - cf_open)
    atr = _calc_atr(candles)

    upper_wick = cf_high - max(cf_open, cf_close)
    lower_wick = min(cf_open, cf_close) - cf_low

    rsi = _calc_rsi(closes, 14)
    pct_b = _calc_bollinger_pct_b(closes, 14)
    stoch_k, stoch_d = _calc_stochastic(candles, 14, 3)
    _, _, m_hist = _calc_macd(closes, 7, 14, 5)
    cci = _calc_cci(candles, 14)
    adx = _calc_adx(candles, 14)

    is_flat_market = adx < 12.0 and abs(cf_close - float(prev_c.close)) < 0.10 * atr and body < 0.10 * atr
    if is_flat_market:
        return "SKIP", "mercado_lateral_sem_volatilidade"

    is_candle_bearish = cf_close < cf_open or cf_close < float(prev_c.close)
    is_candle_bullish = cf_close > cf_open or cf_close > float(prev_c.close)
    is_doji = body < 0.08 * atr

    if signal_side == "CALL":
        is_extreme_overbought = pct_b >= 0.92 or rsi >= 70.0
        has_heavy_top_rejection = not is_doji and upper_wick > 0.80 * body and upper_wick > 0.35 * atr
        if is_extreme_overbought:
            return "SKIP", "veto_exaustao_sobrecompra"
        if has_heavy_top_rejection:
            return "SKIP", "veto_rejeicao_topo"
        if is_doji and lower_wick <= upper_wick:
            return "SKIP", "veto_indecisao_doji"
        if m_hist < 0.0 and cci < -100.0 and pct_b < 0.25:
            return "SKIP", "veto_pressao_vendedora_excessiva"
        has_bottom_support = lower_wick > 0.60 * body or lower_wick > 0.30 * atr
        if has_bottom_support or (stoch_k >= stoch_d and not is_candle_bearish):
            return "CALL", "pullback_suporte_retomada"
        return "CALL", "fluxo_tendencia_saudavel"

    if signal_side == "PUT":
        is_extreme_oversold = pct_b <= 0.08 or rsi <= 30.0
        has_heavy_bottom_rejection = not is_doji and lower_wick > 0.80 * body and lower_wick > 0.35 * atr
        if is_extreme_oversold:
            return "SKIP", "veto_exaustao_sobrevenda"
        if has_heavy_bottom_rejection:
            return "SKIP", "veto_rejeicao_fundo"
        if is_doji and upper_wick <= lower_wick:
            return "SKIP", "veto_indecisao_doji"
        if m_hist > 0.0 and cci > 100.0 and pct_b > 0.75:
            return "SKIP", "veto_pressao_compradora_excessiva"
        has_top_resistance = upper_wick > 0.60 * body or upper_wick > 0.30 * atr
        if has_top_resistance or (stoch_k <= stoch_d and not is_candle_bullish):
            return "PUT", "pullback_resistencia_retomada"
        return "PUT", "fluxo_queda_saudavel"

    return signal_side, "direcao_preservada"

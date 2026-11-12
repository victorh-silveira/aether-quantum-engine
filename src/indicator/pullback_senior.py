"""Avaliacao institucional senior de tendencia, recuo, FVG e killzones."""

from __future__ import annotations

import sys
from typing import Any

from indicator.domain import Candle
from indicator.institutional import detect_fvg_mitigation, is_within_killzone, session_price_reference
from indicator.pullback_indicators import (
    calc_adx,
    calc_bollinger_pct_b,
    calc_hurst_exponent,
    calc_macd,
    calc_rsi,
    calc_stochastic,
    calc_williams_r,
)


def evaluate_senior_institutional(
    bars: list[Candle],
    closes: list[float],
    fast: list[float],
    slow: list[float],
    atr: float,
    optimal_hours_brt: list[int] | None = None,
    module: Any = None,
) -> tuple[str, str]:
    """Avalia setup institucional senior sob confluencia dos 5 pilares."""
    mod = module or sys.modules.get("indicator.pullback")

    def _get(name: str, fallback: Any) -> Any:
        """Obtem funcao injetada dinamicamente com fallback estatico."""
        return getattr(mod, name, fallback) if mod is not None else fallback

    pullback, confirmation = bars[-2:]
    if any(bar.high == bar.low for bar in bars[-3:]):
        return "SEM SINAL", "mercado_sem_oscilacao"
    bar_period = confirmation.epoch - pullback.epoch
    killzone_period = bar_period if bar_period in (300, 900, 3600, 86400) else 0
    if not is_within_killzone(confirmation.epoch, optimal_hours_brt, period_seconds=killzone_period):
        return "SEM SINAL", "fora_da_killzone_brt"

    fn_ref = _get("session_price_reference", session_price_reference)
    fn_fvg = _get("detect_fvg_mitigation", detect_fvg_mitigation)
    fn_adx = _get("_adx", calc_adx)
    fn_macd = _get("_macd", calc_macd)
    fn_rsi = _get("_rsi", calc_rsi)
    fn_stoch = _get("_stochastic", calc_stochastic)
    fn_pct_b = _get("_bollinger_pct_b", calc_bollinger_pct_b)
    fn_williams = _get("_williams_r", calc_williams_r)
    fn_hurst = _get("_hurst_exponent", calc_hurst_exponent)

    reference, upper, lower = fn_ref(bars)
    bull_fvg, bear_fvg = fn_fvg(bars, atr)
    body = abs(confirmation.close - confirmation.open)
    candle_range = max(1e-10, confirmation.high - confirmation.low)
    bounded = body <= 2.5 * atr and candle_range <= 2.60 * atr
    is_doji = body < 0.025 * atr
    adx, pdi, mdi = fn_adx(bars, 14)
    m_val, s_val, m_hist = fn_macd(closes, 7, 14, 5)
    _, _, m_hist_prev = fn_macd(closes[:-1], 7, 14, 5)
    rsi = fn_rsi(closes, 14)
    rsi_prev = fn_rsi(closes[:-1], 14)
    stoch_k, _ = fn_stoch(bars, 14, 3)
    pct_b = fn_pct_b(closes, 14)
    williams = fn_williams(bars, 14)
    hurst = fn_hurst(closes, 14)

    upper_wick = confirmation.high - max(confirmation.open, confirmation.close)
    lower_wick = min(confirmation.open, confirmation.close) - confirmation.low
    call_rejection = upper_wick >= 0.50 * candle_range and upper_wick >= 0.40 * atr and upper_wick >= 1.1 * body
    put_rejection = lower_wick >= 0.50 * candle_range and lower_wick >= 0.40 * atr and lower_wick >= 1.1 * body

    has_variance = len(set(closes[-16:-2])) > 2 if len(closes) >= 18 else False

    streak_bull = (
        len(bars) >= 3
        and bars[-3].close > bars[-3].open
        and bars[-2].close > bars[-2].open
        and bars[-1].close > bars[-1].open
    )
    streak_bear = (
        len(bars) >= 3
        and bars[-3].close < bars[-3].open
        and bars[-2].close < bars[-2].open
        and bars[-1].close < bars[-1].open
    )

    trend_strong = (adx == 0.0 and pdi == 0.0 and mdi == 0.0) or adx >= 8.0
    is_high_momentum = trend_strong and adx >= 26.0 and hurst >= 0.50

    macd_bull_div = m_hist > 0.0 and m_val > s_val
    macd_bear_div = m_hist < 0.0 and m_val < s_val

    climax_call = stoch_k >= 90.0 or williams >= -8.0 or (macd_bear_div and confirmation.close > fast[-1] + 0.80 * atr)
    climax_put = stoch_k <= 10.0 or williams <= -92.0 or (macd_bull_div and confirmation.close < fast[-1] - 0.80 * atr)
    rsi_bear_div = confirmation.close > pullback.close and rsi < rsi_prev - 3.5
    rsi_bull_div = confirmation.close < pullback.close and rsi > rsi_prev + 3.5

    call_exhaustion = has_variance and (
        call_rejection
        or hurst < 0.38
        or climax_call
        or rsi_bear_div
        or rsi >= 75.0
        or stoch_k >= 82.0
        or (not is_high_momentum and (rsi >= 68.0 or pct_b >= 0.88 or williams >= -12.0 or streak_bull or hurst < 0.40))
        or (is_high_momentum and (rsi >= 75.0 or pct_b >= 1.05))
    )
    put_exhaustion = has_variance and (
        put_rejection
        or hurst < 0.38
        or climax_put
        or rsi_bull_div
        or rsi <= 25.0
        or stoch_k <= 18.0
        or (not is_high_momentum and (rsi <= 32.0 or pct_b <= 0.12 or williams <= -88.0 or streak_bear or hurst < 0.40))
        or (is_high_momentum and (rsi <= 25.0 or pct_b <= -0.05))
    )
    call_dmi_ok = (pdi == 0.0 and mdi == 0.0) or pdi >= mdi * 0.55
    put_dmi_ok = (pdi == 0.0 and mdi == 0.0) or mdi >= pdi * 0.55
    call_macd_ok = (m_hist >= -0.25 * atr) or (m_hist > m_hist_prev and m_hist >= -0.40 * atr)
    put_macd_ok = (m_hist <= 0.25 * atr) or (m_hist < m_hist_prev and m_hist <= 0.40 * atr)
    fast_bear_cross = (
        fast[-1] < slow[-1]
        or fast[-2] < slow[-2]
        or (confirmation.close < fast[-1] and fast[-1] < fast[-2] and mdi > pdi)
    )
    fast_bull_cross = (
        fast[-1] > slow[-1]
        or fast[-2] > slow[-2]
        or (confirmation.close > fast[-1] and fast[-1] > fast[-2] and pdi > mdi)
    )
    structural_call = (
        trend_strong
        and call_dmi_ok
        and call_macd_ok
        and not is_doji
        and fast_bull_cross
        and fast[-1] >= fast[-3] - 0.20 * atr
        and slow[-1] >= slow[-4] - 0.05 * atr
        and confirmation.close > slow[-1]
    )
    structural_put = (
        trend_strong
        and put_dmi_ok
        and put_macd_ok
        and not is_doji
        and fast_bear_cross
        and fast[-1] <= fast[-3] + 0.20 * atr
        and slow[-1] <= slow[-4] + 0.05 * atr
        and confirmation.close < slow[-1]
    )
    trend_call = structural_call and not call_exhaustion
    trend_put = structural_put and not put_exhaustion
    pb_mid = (pullback.open + pullback.close) / 2.0
    is_classic_bull_pullback = pullback.close < pullback.open and (
        confirmation.close > pullback.high or confirmation.close >= pb_mid
    )
    is_classic_bear_pullback = pullback.close > pullback.open and (
        confirmation.close < pullback.low or confirmation.close <= pb_mid
    )
    ref_call_ok = reference <= 0 or (
        (lower <= 0 or confirmation.close >= lower - 0.50 * atr)
        and (upper <= 0 or confirmation.close <= upper + 1.20 * atr)
    )
    ref_put_ok = reference <= 0 or (
        (upper <= 0 or confirmation.close <= upper + 0.50 * atr)
        and (lower <= 0 or confirmation.close >= lower - 1.20 * atr)
    )
    max_dist_fast = 1.75 * atr if is_high_momentum else 1.50 * atr
    max_pb_dist = 2.50 * atr if is_high_momentum else 2.00 * atr
    call_pullback = (
        trend_call
        and abs(pullback.low - fast[-2]) <= max_pb_dist
        and is_classic_bull_pullback
        and confirmation.close > confirmation.open
        and abs(confirmation.close - fast[-1]) <= max_dist_fast
        and bounded
        and ref_call_ok
        and bull_fvg
    )
    call_continuation = (
        trend_call
        and not is_classic_bull_pullback
        and (bull_fvg or abs(pullback.low - fast[-2]) <= max_pb_dist)
        and abs(confirmation.close - fast[-1]) <= max_dist_fast
        and bounded
        and ref_call_ok
        and confirmation.close > confirmation.open
    )
    put_pullback = (
        trend_put
        and abs(pullback.high - fast[-2]) <= max_pb_dist
        and is_classic_bear_pullback
        and confirmation.close < confirmation.open
        and abs(confirmation.close - fast[-1]) <= max_dist_fast
        and bounded
        and ref_put_ok
        and bear_fvg
    )
    put_continuation = (
        trend_put
        and not is_classic_bear_pullback
        and (bear_fvg or abs(pullback.high - fast[-2]) <= max_pb_dist)
        and abs(confirmation.close - fast[-1]) <= max_dist_fast
        and bounded
        and ref_put_ok
        and confirmation.close < confirmation.open
    )
    call = call_pullback or call_continuation
    put = put_pullback or put_continuation
    side = "CALL" if call else "PUT" if put else "SEM SINAL"
    if side != "SEM SINAL":
        reason = "setup_confirmado"
    elif (structural_call and call_rejection) or (structural_put and put_rejection):
        reason = "rejeicao_pavio_contrario"
    elif not structural_call and not structural_put:
        reason = "sem_tendencia_alinhada"
    elif abs(confirmation.close - fast[-1]) > max_dist_fast:
        reason = "afastado_da_media"
    elif (structural_call and confirmation.close <= confirmation.open) or (
        structural_put and confirmation.close >= confirmation.open
    ):
        reason = "vela_contraria"
    elif (structural_call and call_exhaustion) or (structural_put and put_exhaustion):
        reason = "climax_exaustao"
    else:
        reason = "padrao_incompleto"
    return side, reason

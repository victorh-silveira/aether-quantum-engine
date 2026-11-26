"""Indicadores tecnicos causais unificados em velas fechadas (Polars/Python puro)."""

from __future__ import annotations

from math import log

from src.domain.indicators.snapshot import IndicatorSnapshot
from src.domain.models.market_data import Candle


__all__ = ["IndicatorSnapshot", "compute_indicators"]


def _ema_series(values: list[float], span: int) -> list[float]:
    """Calcula serie EMA causal direta."""
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    out = [values[0]]
    for val in values[1:]:
        out.append(alpha * val + (1.0 - alpha) * out[-1])
    return out


def _true_range(curr: Candle, prev: Candle) -> float:
    """Calcula True Range entre a vela atual e o fechamento anterior."""
    return max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close))


def _wilder_atr(bars: list[Candle], period: int) -> float:
    """Calcula Average True Range causal pelo metodo Wilder."""
    if len(bars) < 2 or period <= 0:
        return 0.0
    trs = [_true_range(curr, prev) for prev, curr in zip(bars[:-1], bars[1:], strict=True)]
    if len(trs) < period:
        return sum(trs) / len(trs)
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def _wilder_adx(bars: list[Candle], period: int) -> tuple[float, float, float, float]:
    """Calcula (ADX, +DI, -DI, DI_diff) causais pelo metodo Wilder EWM."""
    if len(bars) <= period or period <= 0:
        return 0.0, 0.0, 0.0, 0.0
    alpha = 1.0 / period
    one_minus = 1.0 - alpha
    ewm_tr = max(bars[0].high - bars[0].low, 1e-12)
    ewm_pdm = 0.0
    ewm_mdm = 0.0
    ewm_dx = 0.0
    for prev, cur in zip(bars[:-1], bars[1:], strict=True):
        tr = _true_range(cur, prev)
        up = cur.high - prev.high
        dn = prev.low - cur.low
        pdm = up if (up > dn and up > 0.0) else 0.0
        mdm = dn if (dn > up and dn > 0.0) else 0.0
        ewm_tr = one_minus * ewm_tr + alpha * tr
        ewm_pdm = one_minus * ewm_pdm + alpha * pdm
        ewm_mdm = one_minus * ewm_mdm + alpha * mdm
        if ewm_tr > 1e-12:
            pdi = 100.0 * ewm_pdm / ewm_tr
            mdi = 100.0 * ewm_mdm / ewm_tr
            di_sum = pdi + mdi
            dx = (100.0 * abs(pdi - mdi) / di_sum) if di_sum > 1e-12 else 0.0
            ewm_dx = one_minus * ewm_dx + alpha * dx
    if ewm_tr <= 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    pdi = 100.0 * ewm_pdm / ewm_tr
    mdi = 100.0 * ewm_mdm / ewm_tr
    return ewm_dx, pdi, mdi, pdi - mdi


def _calc_rsi(closes: list[float], period: int) -> float:
    """Calcula RSI causal no periodo informado."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [curr - prev for prev, curr in zip(closes[-period - 1 : -1], closes[-period:], strict=True)]
    avg_g = sum(max(0.0, d) for d in deltas) / period
    avg_l = sum(max(0.0, -d) for d in deltas) / period
    if avg_l <= 1e-12:
        return 100.0 if avg_g > 0.0 else 50.0
    return 100.0 - (100.0 / (1.0 + avg_g / avg_l))


def _calc_macd(closes: list[float]) -> tuple[float, float, float, float]:
    """Calcula MACD (12, 26, 9) e variacao do histograma."""
    if len(closes) < 35:
        return 0.0, 0.0, 0.0, 0.0
    fast = _ema_series(closes, 12)
    slow = _ema_series(closes, 26)
    line = [f - s for f, s in zip(fast, slow, strict=True)]
    sig = _ema_series(line, 9)
    hist = line[-1] - sig[-1]
    hist_prev = line[-2] - sig[-2]
    return line[-1], sig[-1], hist, hist - hist_prev


def _calc_stochastic(bars: list[Candle], period: int = 14, smooth: int = 3) -> tuple[float, float]:
    """Calcula %K e %D do Estocastico causal."""
    if len(bars) < period + smooth:
        return 50.0, 50.0
    ks: list[float] = []
    for idx in range(len(bars) - smooth + 1, len(bars) + 1):
        win = bars[idx - period : idx]
        hh = max(b.high for b in win)
        ll = min(b.low for b in win)
        rng = hh - ll
        ks.append(((win[-1].close - ll) / rng * 100.0) if rng > 1e-12 else 50.0)
    return ks[-1], sum(ks) / len(ks)


def _calc_cci_cmo_wr_roc(
    bars: list[Candle], closes: list[float], period: int = 14
) -> tuple[float, float, float, float]:
    """Calcula CCI, CMO, Williams %R e ROC causais."""
    cci = 0.0
    wr = -50.0
    if len(bars) >= period:
        win = bars[-period:]
        tps = [(b.high + b.low + b.close) / 3.0 for b in win]
        mean_tp = sum(tps) / period
        mdev = sum(abs(tp - mean_tp) for tp in tps) / period
        cci = ((tps[-1] - mean_tp) / (0.015 * mdev)) if mdev > 1e-10 else 0.0
        hh = max(b.high for b in win)
        ll = min(b.low for b in win)
        rng = hh - ll
        wr = (((hh - win[-1].close) / rng) * -100.0) if rng > 1e-10 else -50.0
    cmo = 0.0
    roc = 0.0
    if len(closes) >= period + 1:
        deltas = [c - p for p, c in zip(closes[-period - 1 : -1], closes[-period:], strict=True)]
        gains = sum(d for d in deltas if d > 0.0)
        losses = sum(-d for d in deltas if d < 0.0)
        denom = gains + losses
        cmo = (((gains - losses) / denom) * 100.0) if denom > 1e-10 else 0.0
        base = closes[-period - 1]
        roc = (((closes[-1] - base) / base) * 100.0) if abs(base) > 1e-10 else 0.0
    return cci, cmo, wr, roc


def _calc_bands_and_hurst(closes: list[float], ema20: float, atr14: float) -> tuple[float, float, float, float]:
    """Calcula Bollinger %B, largura BB, Keltner %B e Expoente de Hurst."""
    bb_b, bb_w = 0.5, 0.0
    if len(closes) >= 20:
        win = closes[-20:]
        mean = sum(win) / 20.0
        std = (sum((x - mean) ** 2 for x in win) / 20.0) ** 0.5
        width = 4.0 * std
        bb_b = (win[-1] - (mean - 2.0 * std)) / width if width > 1e-12 else 0.5
        bb_w = width / mean if abs(mean) > 1e-12 else 0.0
    kc_width = 4.0 * atr14
    kc_b = (closes[-1] - (ema20 - 2.0 * atr14)) / kc_width if kc_width > 1e-12 else 0.5
    hurst = 0.5
    if len(closes) >= 14:
        ser = closes[-14:]
        mean = sum(ser) / 14.0
        devs = [x - mean for x in ser]
        cum, acc = [], 0.0
        for d in devs:
            acc += d
            cum.append(acc)
        r = max(cum) - min(cum)
        s = (sum(d * d for d in devs) / 14.0) ** 0.5
        if r > 1e-10 and s > 1e-10:
            hurst = min(1.0, max(0.0, float(log(r / s) / log(14.0))))
    return bb_b, bb_w, kc_b, hurst


def _calc_microstructure(bars: list[Candle], atr14: float) -> tuple[float, float, float, int, int, float]:
    """Calcula corpo/pavios normalizados, direcao, streak e z-score da sessao UTC."""
    last = bars[-1]
    norm = atr14 if atr14 > 1e-12 else max(last.high - last.low, 1e-12)
    body_atr = abs(last.close - last.open) / norm
    upper_atr = (last.high - max(last.open, last.close)) / norm
    lower_atr = (min(last.open, last.close) - last.low) / norm
    c_dir = 1 if last.close > last.open else (-1 if last.close < last.open else 0)
    streak = 0
    if c_dir != 0:
        for bar in reversed(bars):
            b_dir = 1 if bar.close > bar.open else (-1 if bar.close < bar.open else 0)
            if b_dir != c_dir:
                break
            streak += c_dir
    day_start = (last.epoch // 86400) * 86400
    sess_tps = [(b.high + b.low + b.close) / 3.0 for b in bars if b.epoch >= day_start]
    if len(sess_tps) >= 2:
        mean_s = sum(sess_tps) / len(sess_tps)
        std_s = (sum((x - mean_s) ** 2 for x in sess_tps) / len(sess_tps)) ** 0.5
        zscore = (last.close - mean_s) / std_s if std_s > 1e-12 else 0.0
    else:
        zscore = 0.0
    return body_atr, upper_atr, lower_atr, c_dir, streak, zscore


def compute_indicators(bars: list[Candle]) -> IndicatorSnapshot:
    """Calcula o pacote completo de indicadores causais sobre velas M5 fechadas."""
    if not bars:
        raise ValueError("lista de velas vazia para calculo de indicadores")
    closes = [b.close for b in bars]
    ema9_s = _ema_series(closes, 9)
    ema20_s = _ema_series(closes, 20)
    ema21_s = _ema_series(closes, 21)
    ema50_s = _ema_series(closes, 50)
    atr14 = _wilder_atr(bars, 14)
    atr50 = _wilder_atr(bars, 50)
    atr_norm = atr14 if atr14 > 1e-12 else 1.0
    ema20_slope = (ema20_s[-1] - ema20_s[-4]) / atr_norm if len(ema20_s) >= 4 else 0.0
    ema50_slope = (ema50_s[-1] - ema50_s[-4]) / atr_norm if len(ema50_s) >= 4 else 0.0
    macd, macd_sig, macd_hist, macd_delta = _calc_macd(closes)
    stoch_k, stoch_d = _calc_stochastic(bars)
    cci14, cmo14, wr14, roc14 = _calc_cci_cmo_wr_roc(bars, closes)
    adx14, pdi14, mdi14, di_diff14 = _wilder_adx(bars, 14)
    adx20, _, _, di_diff20 = _wilder_adx(bars, 20)
    bb_b, bb_w, kc_b, hurst14 = _calc_bands_and_hurst(closes, ema20_s[-1], atr14)
    body_atr, up_atr, lo_atr, c_dir, streak, zscore = _calc_microstructure(bars, atr14)
    return IndicatorSnapshot(
        close=closes[-1],
        ema_9=ema9_s[-1],
        ema_20=ema20_s[-1],
        ema_21=ema21_s[-1],
        ema_50=ema50_s[-1],
        ema_spread_atr=(ema20_s[-1] - ema50_s[-1]) / atr_norm if atr14 > 1e-12 else 0.0,
        ema20_slope=ema20_slope,
        ema50_slope=ema50_slope,
        rsi_7=_calc_rsi(closes, 7),
        rsi_14=_calc_rsi(closes, 14),
        macd=macd,
        macd_signal=macd_sig,
        macd_hist=macd_hist,
        macd_hist_delta=macd_delta,
        stoch_k=stoch_k,
        stoch_d=stoch_d,
        cci_14=cci14,
        cmo_14=cmo14,
        williams_r_14=wr14,
        roc_14=roc14,
        adx_14=adx14,
        plus_di_14=pdi14,
        minus_di_14=mdi14,
        di_diff_14=di_diff14,
        adx_20=adx20,
        di_diff_20=di_diff20,
        atr_14=atr14,
        atr_50=atr50,
        atr_ratio=(atr14 / atr50) if atr50 > 1e-12 else 1.0,
        bollinger_pct_b=bb_b,
        bollinger_width=bb_w,
        keltner_pct_b=kc_b,
        hurst_14=hurst14,
        body_atr=body_atr,
        upper_wick_atr=up_atr,
        lower_wick_atr=lo_atr,
        candle_dir=c_dir,
        streak=streak,
        session_zscore=zscore,
    )

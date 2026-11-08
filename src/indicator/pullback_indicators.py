"""Funcoes causais de indicadores matematicos e estatisticos para calculo do pullback."""

from __future__ import annotations

from math import log

from indicator.domain import Candle


def ema_direct(values: list[float], span: int) -> list[float]:
    """Calcula media movel exponencial causal direta."""
    alpha = 2.0 / (span + 1.0)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1.0 - alpha) * result[-1])
    return result


def calc_ema(values: list[float], span: int) -> list[float]:
    """Calcula media movel exponencial."""
    return ema_direct(values, span)


def calc_atr(candles: list[Candle]) -> float:
    """Calcula Average True Range (ATR) causal de 14 periodos."""
    ranges = []
    for previous, current in zip(candles[-15:-1], candles[-14:], strict=True):
        ranges.append(
            max(current.high - current.low, abs(current.high - previous.close), abs(current.low - previous.close))
        )
    return sum(ranges) / 14.0


def calc_rsi(closes: list[float], period: int = 14) -> float:
    """Calcula o RSI causal de 14 periodos."""
    if len(closes) < period + 1:
        return 50.0
    gains = []
    losses = []
    for prev, curr in zip(closes[-period - 1 : -1], closes[-period:], strict=True):
        chg = curr - prev
        gains.append(max(0.0, chg))
        losses.append(max(0.0, -chg))
    avg_g = sum(gains) / period
    avg_l = sum(losses) / period
    if avg_l == 0:
        return 100.0 if avg_g > 0 else 50.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))


def calc_bollinger_pct_b(closes: list[float], period: int = 14, mult: float = 2.0) -> float:
    """Calcula o %B das Bandas de Bollinger causal de 14 periodos."""
    if len(closes) < period:
        return 0.5
    recent = closes[-period:]
    mean = sum(recent) / period
    variance = sum((x - mean) ** 2 for x in recent) / period
    std = variance**0.5
    upper = mean + mult * std
    lower = mean - mult * std
    width = upper - lower
    return (recent[-1] - lower) / width if width > 0 else 0.5


def calc_adx(bars: list[Candle], period: int = 14) -> tuple[float, float, float]:
    """Calcula ADX e direcionais +DI / -DI causais de 14 periodos."""
    if len(bars) < period + 1:
        return 0.0, 0.0, 0.0
    tr_sum = 0.0
    pdm_sum = 0.0
    mdm_sum = 0.0
    for prev, curr in zip(bars[-period - 1 : -1], bars[-period:], strict=True):
        tr = max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close))
        tr_sum += tr
        up = curr.high - prev.high
        down = prev.low - curr.low
        if up > down and up > 0:
            pdm_sum += up
        if down > up and down > 0:
            mdm_sum += down
    if tr_sum <= 0:
        return 0.0, 0.0, 0.0
    plus_di = (pdm_sum / tr_sum) * 100.0
    minus_di = (mdm_sum / tr_sum) * 100.0
    di_sum = plus_di + minus_di
    dx = (abs(plus_di - minus_di) / di_sum * 100.0) if di_sum > 0 else 0.0
    return dx, plus_di, minus_di


def calc_macd(
    closes: list[float], fast_span: int = 7, slow_span: int = 14, signal_span: int = 5
) -> tuple[float, float, float]:
    """Calcula linha MACD, sinal e histograma causais harmonizados com 14 periodos."""
    if len(closes) < slow_span + signal_span:
        return 0.0, 0.0, 0.0
    fast = ema_direct(closes, fast_span)
    slow = ema_direct(closes, slow_span)
    macd_line = [f - s for f, s in zip(fast, slow, strict=True)]
    sig_line = ema_direct(macd_line, signal_span)
    m_val = macd_line[-1]
    s_val = sig_line[-1]
    hist = m_val - s_val
    return m_val, s_val, hist


def calc_stochastic(bars: list[Candle], period: int = 14, smooth: int = 3) -> tuple[float, float]:
    """Calcula %K e %D do Estocastico causal."""
    if len(bars) < period + smooth:
        return 50.0, 50.0
    k_vals = []
    for idx in range(len(bars) - smooth + 1, len(bars) + 1):
        win = bars[idx - period : idx]
        high_max = max(b.high for b in win)
        low_min = min(b.low for b in win)
        rng = high_max - low_min
        close = win[-1].close
        k = ((close - low_min) / rng * 100.0) if rng > 0 else 50.0
        k_vals.append(k)
    pct_k = k_vals[-1]
    pct_d = sum(k_vals) / len(k_vals) if k_vals else 50.0
    return pct_k, pct_d


def calc_cci(bars: list[Candle], period: int = 14, constant: float = 0.015) -> float:
    """Calcula Commodity Channel Index (CCI) causal de 14 periodos."""
    if len(bars) < period:
        return 0.0
    recent = bars[-period:]
    tp_list = [(b.high + b.low + b.close) / 3.0 for b in recent]
    mean_tp = sum(tp_list) / period
    mean_dev = sum(abs(tp - mean_tp) for tp in tp_list) / period
    return ((tp_list[-1] - mean_tp) / (constant * mean_dev)) if mean_dev > 1e-10 else 0.0


def calc_cmo(closes: list[float], period: int = 14) -> float:
    """Calcula Chande Momentum Oscillator (CMO) causal de 14 periodos."""
    if len(closes) < period + 1:
        return 0.0
    deltas = [curr - prev for prev, curr in zip(closes[-period - 1 : -1], closes[-period:], strict=True)]
    gains = sum(d for d in deltas if d > 0)
    losses = sum(-d for d in deltas if d < 0)
    denom = gains + losses
    return (((gains - losses) / denom) * 100.0) if denom > 1e-10 else 0.0


def calc_williams_r(bars: list[Candle], period: int = 14) -> float:
    """Calcula Williams %R causal de 14 periodos [-100, 0]."""
    if len(bars) < period:
        return -50.0
    recent = bars[-period:]
    hh = max(b.high for b in recent)
    ll = min(b.low for b in recent)
    rng = hh - ll
    return (((hh - recent[-1].close) / rng) * -100.0) if rng > 1e-10 else -50.0


def calc_roc(closes: list[float], period: int = 14) -> float:
    """Calcula Rate of Change (ROC) causal de 14 periodos."""
    if len(closes) < period + 1:
        return 0.0
    base = closes[-period - 1]
    return (((closes[-1] - base) / base) * 100.0) if abs(base) > 1e-10 else 0.0


def calc_keltner_pct_b(bars: list[Candle], period: int = 14, atr_period: int = 14, mult: float = 2.0) -> float:
    """Calcula Keltner Channel %B causal de 14 periodos."""
    if len(bars) < max(period, atr_period) + 1:
        return 0.5
    closes = [b.close for b in bars]
    ema_vals = calc_ema(closes, period)
    ranges = [
        max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close))
        for prev, curr in zip(bars[-atr_period - 1 : -1], bars[-atr_period:], strict=True)
    ]
    atr_kc = (sum(ranges) / atr_period) if ranges else 1.0
    mid = ema_vals[-1]
    upper = mid + mult * atr_kc
    lower = mid - mult * atr_kc
    width = upper - lower
    return ((closes[-1] - lower) / width) if width > 1e-10 else 0.5


def calc_hurst_exponent(closes: list[float], window: int = 14) -> float:
    """Estima o Expoente de Hurst causal de 14 periodos via variancia rescalada."""
    if len(closes) < window:
        return 0.5
    series = closes[-window:]
    mean = sum(series) / window
    deviations = [x - mean for x in series]
    cumdev = []
    curr = 0.0
    for d in deviations:
        curr += d
        cumdev.append(curr)
    r = max(cumdev) - min(cumdev)
    variance = sum(d * d for d in deviations) / window
    s = variance**0.5
    if s <= 1e-10 or r <= 1e-10:
        return 0.5
    h = (log(r / s) / log(window)) if window > 1 else 0.5
    return min(1.0, max(0.0, float(h)))

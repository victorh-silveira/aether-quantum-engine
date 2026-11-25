"""Features em lote com a mesma reinicializacao das janelas escalares de producao."""

from __future__ import annotations

import numpy as np

from src.indicator.domain import Candle
from src.indicator.strategy_persistent_trend import _resolve_h1_trend


WINDOW = 500


def _arrays(bars: list[Candle]) -> tuple[np.ndarray, ...]:
    """Converte velas em vetores numericos sem pandas."""
    return tuple(np.array([getattr(b, key) for b in bars]) for key in ("epoch", "open", "high", "low", "close"))


def rolling_di(bars: list[Candle], period: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduz Wilder de cada janela de ate 500 velas, inclusive o seed local."""
    if not bars:
        return np.zeros(0), np.zeros(0)
    _, _, high, low, close = _arrays(bars)
    ends = np.arange(len(bars))
    starts = np.maximum(0, ends - WINDOW + 1)
    tr = high[starts] - low[starts]
    plus = np.zeros(len(bars))
    minus = np.zeros(len(bars))
    adx = np.zeros(len(bars))
    alpha = 1.0 / period
    decay = 1.0 - alpha
    for offset in range(1, min(WINDOW, len(bars))):
        idx = np.minimum(starts + offset, ends)
        active = starts + offset <= ends
        prev = np.maximum(0, idx - 1)
        true_range = np.maximum(
            high[idx] - low[idx], np.maximum(abs(high[idx] - close[prev]), abs(low[idx] - close[prev]))
        )
        up = high[idx] - high[prev]
        down = low[prev] - low[idx]
        tr = np.where(active, decay * tr + alpha * true_range, tr)
        plus = np.where(active, decay * plus + alpha * np.where((up > down) & (up > 0), up, 0), plus)
        minus = np.where(active, decay * minus + alpha * np.where((down > up) & (down > 0), down, 0), minus)
        pdi = np.divide(100 * plus, tr, out=np.zeros_like(tr), where=tr > 1e-12)
        mdi = np.divide(100 * minus, tr, out=np.zeros_like(tr), where=tr > 1e-12)
        total = pdi + mdi
        dx = np.divide(100 * abs(pdi - mdi), total, out=np.zeros_like(total), where=total > 1e-12)
        adx = np.where(active & (tr > 1e-12), decay * adx + alpha * dx, adx)
    diff = np.divide(100 * (plus - minus), tr, out=np.zeros_like(tr), where=tr > 1e-12)
    enough = ends - starts + 1 > period
    return np.where(enough, diff, 0), np.where(enough & (tr > 1e-12), adx, 0)


def rolling_ema(values: np.ndarray, span: int) -> np.ndarray:
    """EMA com seed no primeiro fechamento de cada janela causal de 500 pontos."""
    ends = np.arange(len(values))
    starts = np.maximum(0, ends - WINDOW + 1)
    result = values[starts].copy()
    alpha = 2.0 / (span + 1.0)
    for offset in range(1, min(WINDOW, len(values))):
        idx = np.minimum(starts + offset, ends)
        result = np.where(starts + offset <= ends, alpha * values[idx] + (1 - alpha) * result, result)
    return result


def batch_features(bars: list[Candle], h1_bars: list[Candle]) -> dict[str, np.ndarray]:
    """Calcula features com corte H1 fechado; runner valida gaps e warmup."""
    epoch, opens, high, low, close = _arrays(bars)
    result = {}
    for period in (14, 20, 24, 28):
        diff, adx = rolling_di(bars, period)
        result[f"di{period}"] = diff
        if period == 20:
            result["adx20"] = adx
    trend = np.zeros(len(bars), dtype=np.int8)
    if h1_bars:
        h_epoch, _, _, _, h_close = _arrays(h1_bars)
        ema20 = rolling_ema(h_close, 20)
        ema50 = rolling_ema(h_close, 50)
        indices = np.searchsorted(h_epoch + 3600, epoch + 900, side="right") - 1
        valid = indices >= 50
        trend[valid] = np.where(ema20[indices[valid]] > ema50[indices[valid]], 1, -1)
    else:
        trend = np.array(
            [_resolve_h1_trend(bars[max(0, i - 499) : i + 1], None) for i in range(len(bars))], dtype=np.int8
        )
    result["h1_trend"] = trend
    result["direction"] = np.sign(close - opens).astype(np.int8)
    flat = high == low
    result["flat"] = flat | np.roll(flat, 1)
    result["toxic"] = np.isin((epoch // 3600) % 24, (3, 6, 19))
    return result

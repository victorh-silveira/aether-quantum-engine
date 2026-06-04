"""Series de preco e tensores de features para o TCN."""

import math

import numpy as np
import polars as pl

from src.application.services.deep_learning.dl_pair_features import (
    pair_feature_row,
    precompute_pair_series,
)


BASE_FEATURE_DIM = 10
DERIV_FEATURE_DIM = 4
PAIR_FEATURE_DIM = 3
FEATURE_DIM = BASE_FEATURE_DIM + DERIV_FEATURE_DIM + PAIR_FEATURE_DIM


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calcula RSI por barra para a serie de precos informada."""
    if len(prices) < period + 1:
        return np.full_like(prices, 50.0)
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / (down + 1e-10)
    rsi = np.zeros_like(prices)
    rsi[:period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            up_val = delta
            down_val = 0.0
        else:
            up_val = 0.0
            down_val = -delta
        up = (up * (period - 1) + up_val) / period
        down = (down * (period - 1) + down_val) / period
        rs = up / (down + 1e-10)
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _bars_per_hour(granularity: int) -> int:
    """Quantidade de barras em uma hora para a granularidade informada."""
    gran = max(60, int(granularity))
    return max(1, 3600 // gran)


def _attach_deriv_ohlc(
    series: dict[str, np.ndarray],
    prices: np.ndarray,
    *,
    open_: np.ndarray | None,
    high: np.ndarray | None,
    low: np.ndarray | None,
) -> None:
    """Anexa body, range, close_loc e upper_wick derivados do OHLC Deriv."""
    n = len(prices)
    body = np.zeros(n, dtype=np.float64)
    range_pct = np.zeros(n, dtype=np.float64)
    close_loc = np.zeros(n, dtype=np.float64)
    upper_wick = np.zeros(n, dtype=np.float64)
    if (
        open_ is not None
        and high is not None
        and low is not None
        and len(open_) >= n
        and len(high) >= n
        and len(low) >= n
    ):
        o = np.asarray(open_[:n], dtype=np.float64)
        h = np.asarray(high[:n], dtype=np.float64)
        low_px = np.asarray(low[:n], dtype=np.float64)
        c = np.asarray(prices[:n], dtype=np.float64)
        body = (c - o) / (o + 1e-10)
        range_pct = (h - low_px) / (o + 1e-10)
        close_loc = (c - low_px) / (h - low_px + 1e-10)
        upper_wick = (h - np.maximum(o, c)) / (o + 1e-10)
    series["body"] = body
    series["range_pct"] = range_pct
    series["close_loc"] = close_loc
    series["upper_wick"] = upper_wick


def precompute_price_series(
    prices: np.ndarray,
    *,
    granularity: int = 300,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Precomputa series auxiliares usadas na montagem de features."""
    n = len(prices)
    returns = np.zeros(n, dtype=np.float64)
    if n > 1:
        returns[1:] = np.diff(prices) / (prices[:-1] + 1e-10)
    vol = np.zeros(n, dtype=np.float64)
    for i in range(10, n):
        vol[i] = np.std(returns[max(0, i - 10) : i])
    rsi = calculate_rsi(prices) / 100.0
    df = pl.DataFrame({"close": prices})
    ema_fast = df.select(pl.col("close").ewm_mean(span=5)).to_numpy().flatten()
    ema_slow = df.select(pl.col("close").ewm_mean(span=15)).to_numpy().flatten()
    ema_spread = (ema_fast - ema_slow) / (ema_slow + 1e-10)
    sma20 = df.select(pl.col("close").rolling_mean(window_size=20, min_samples=1)).to_numpy().flatten()
    sma_dist = (prices - sma20) / (sma20 + 1e-10)
    ret_5 = np.zeros(n, dtype=np.float64)
    for i in range(5, n):
        ret_5[i] = (prices[i] - prices[i - 5]) / (prices[i - 5] + 1e-10)
    rsi_slope = np.zeros(n, dtype=np.float64)
    if n > 1:
        rsi_slope[1:] = np.diff(rsi)
    rel_vol = np.zeros(n, dtype=np.float64)
    for i in range(20, n):
        base = np.mean(vol[max(0, i - 50) : i + 1]) + 1e-10
        rel_vol[i] = vol[i] / base
    bph = _bars_per_hour(granularity)
    bar_phase = np.array([(i % bph) / float(bph) for i in range(n)], dtype=np.float64)
    bar_sin = np.sin(2.0 * math.pi * bar_phase)
    bar_cos = np.cos(2.0 * math.pi * bar_phase)
    series = {
        "returns": returns,
        "vol": vol,
        "rsi": rsi,
        "ema_spread": ema_spread,
        "sma_dist": sma_dist,
        "ret_5": ret_5,
        "rsi_slope": rsi_slope,
        "rel_vol": rel_vol,
        "bar_sin": bar_sin,
        "bar_cos": bar_cos,
    }
    _attach_deriv_ohlc(series, prices, open_=open_, high=high, low=low)
    return series


def build_feature_row(
    series: dict[str, np.ndarray],
    index: int,
    *,
    pair_series: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Monta vetor de features na barra indicada."""
    base = np.array(
        [
            series["returns"][index - 1] if index > 0 else 0.0,
            series["rsi"][index],
            series["vol"][index],
            series["ema_spread"][index],
            series["ret_5"][index],
            series["sma_dist"][index],
            series["rsi_slope"][index],
            series["rel_vol"][index],
            series["bar_sin"][index],
            series["bar_cos"][index],
        ],
        dtype=np.float32,
    )
    deriv = np.array(
        [
            series["body"][index],
            series["range_pct"][index],
            series["close_loc"][index],
            series["upper_wick"][index],
        ],
        dtype=np.float32,
    )
    if pair_series is None:
        return np.concatenate([base, deriv, np.zeros(PAIR_FEATURE_DIM, dtype=np.float32)], axis=0)
    extra = pair_feature_row(pair_series, index)
    return np.concatenate([base, deriv, extra], axis=0).astype(np.float32)


def build_sequence_tensor(
    prices: np.ndarray,
    lookback: int,
    end_index: int,
    *,
    granularity: int = 300,
    pair_prices: np.ndarray | None = None,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
) -> np.ndarray:
    """Monta janela (lookback, FEATURE_DIM) terminando em end_index inclusive."""
    series = precompute_price_series(
        prices,
        granularity=granularity,
        open_=open_,
        high=high,
        low=low,
    )
    pair_series = (
        precompute_pair_series(prices, pair_prices) if pair_prices is not None and len(pair_prices) > 0 else None
    )
    start = end_index - lookback + 1
    rows = [build_feature_row(series, i, pair_series=pair_series) for i in range(start, end_index + 1)]
    return np.stack(rows, axis=0).astype(np.float32)

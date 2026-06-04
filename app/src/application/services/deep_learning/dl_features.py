"""Extracao de features sequenciais e meta-labels para treino TCN."""

import math

import numpy as np
import polars as pl

from src.application.services.deep_learning.dl_pair_features import (
    pair_feature_row,
    precompute_pair_series,
    spread_confirms_direction,
)


BASE_FEATURE_DIM = 10
PAIR_FEATURE_DIM = 3
FEATURE_DIM = BASE_FEATURE_DIM + PAIR_FEATURE_DIM


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


def precompute_price_series(prices: np.ndarray, *, granularity: int = 300) -> dict[str, np.ndarray]:
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
    return {
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
    if pair_series is None:
        return np.pad(base, (0, PAIR_FEATURE_DIM), mode="constant")
    extra = pair_feature_row(pair_series, index)
    return np.concatenate([base, extra], axis=0).astype(np.float32)


def build_sequence_tensor(
    prices: np.ndarray,
    lookback: int,
    end_index: int,
    *,
    granularity: int = 300,
    pair_prices: np.ndarray | None = None,
) -> np.ndarray:
    """Monta janela (lookback, FEATURE_DIM) terminando em end_index inclusive."""
    series = precompute_price_series(prices, granularity=granularity)
    pair_series = (
        precompute_pair_series(prices, pair_prices) if pair_prices is not None and len(pair_prices) > 0 else None
    )
    start = end_index - lookback + 1
    rows = [build_feature_row(series, i, pair_series=pair_series) for i in range(start, end_index + 1)]
    return np.stack(rows, axis=0).astype(np.float32)


def extract_sequences(
    prices: np.ndarray,
    lookback: int,
    *,
    label_min_move_pct: float = 0.0002,
    granularity: int = 300,
    pair_prices: np.ndarray | None = None,
    require_pair_label: bool = False,
    sym_is_bull: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrai tensores (N, L, F), rotulos binarios e mascara meta-label."""
    n = len(prices)
    if n < lookback + 5:
        return np.empty((0, lookback, FEATURE_DIM)), np.empty((0,)), np.empty((0,))
    threshold = max(0.0, float(label_min_move_pct))
    sequences = []
    targets = []
    masks = []
    for i in range(lookback, n - 1):
        move = abs(prices[i + 1] - prices[i]) / (prices[i] + 1e-10)
        target_up = prices[i + 1] > prices[i]
        pair_ok = True
        if require_pair_label and pair_prices is not None and len(pair_prices) >= n:
            pair_ok = spread_confirms_direction(
                prices,
                pair_prices,
                i,
                target_up=target_up,
                sym_is_bull=sym_is_bull,
            )
        active = move >= threshold and pair_ok
        sequences.append(
            build_sequence_tensor(
                prices,
                lookback,
                i,
                granularity=granularity,
                pair_prices=pair_prices,
            )
        )
        targets.append(1.0 if target_up else 0.0)
        masks.append(1.0 if active else 0.0)
    return (
        np.array(sequences, dtype=np.float32),
        np.array(targets, dtype=np.float32),
        np.array(masks, dtype=np.float32),
    )


def extract_features(prices: np.ndarray, lookback: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Compatibilidade: retorna ultima linha de cada sequencia como matriz 2D."""
    seqs, targets, _ = extract_sequences(prices, lookback)
    if len(seqs) == 0:
        return np.empty((0, FEATURE_DIM)), np.empty((0,))
    flat = seqs[:, -1, :]
    return flat, targets

"""Indicadores tecnicos normalizados para features DL Rise/Fall."""

import numpy as np
import polars as pl

from src.application.services.deep_learning.dl_feature_indicators_advanced import (
    calculate_adx,
    calculate_ema_crossover,
    calculate_volatility_ratio,
    calculate_williams_r,
)


__all__ = [
    "calculate_rsi",
    "feature_windows",
    "bollinger",
    "atr_norm",
    "log_returns",
    "rate_of_change",
    "delta_series",
    "rolling_realized_vol_ratio",
    "price_zscore",
    "ema_distances",
    "calculate_macd",
    "calculate_stochastic",
    "calculate_cci",
    "calculate_adx",
    "calculate_williams_r",
    "calculate_volatility_ratio",
    "calculate_ema_crossover",
]


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


def feature_windows(granularity: int) -> dict[str, int]:
    """Janelas de indicadores para contratos alinhados a barras de 60s."""
    _ = max(1, int(granularity))
    return {
        "adx_period": 14,
        "atr_window": 14,
        "bb_window": 20,
        "cci_period": 20,
        "ema_20": 20,
        "ema_50": 50,
        "ema_fast_crossover": 9,
        "ema_slow_crossover": 21,
        "hurst_window": 64,
        "macd_fast": 12,
        "macd_signal": 9,
        "macd_slow": 26,
        "rel_vol_span": 50,
        "roc_period": 10,
        "rsi_period": 14,
        "stoch_period": 14,
        "stoch_smooth": 3,
        "vol_ratio_long": 20,
        "vol_ratio_short": 5,
        "vol_window": 10,
        "williams_period": 14,
        "vr_long": 8,
        "vr_short": 2,
    }


def bollinger(prices: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna banda inferior, media e superior."""
    n = len(prices)
    lower = np.zeros(n, dtype=np.float64)
    mid = np.zeros(n, dtype=np.float64)
    upper = np.zeros(n, dtype=np.float64)
    w = max(2, int(window))
    for i in range(n):
        start = max(0, i - w + 1)
        segment = prices[start : i + 1]
        m = float(np.mean(segment))
        s = float(np.std(segment))
        mid[i] = m
        lower[i] = m - 2.0 * s
        upper[i] = m + 2.0 * s
    return lower, mid, upper


def atr_norm(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """ATR normalizado pelo preco."""
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    out = np.zeros(n, dtype=np.float64)
    w = max(2, int(window))
    for i in range(n):
        start = max(0, i - w + 1)
        out[i] = float(np.mean(tr[start : i + 1])) / (close[i] + 1e-10)
    return out


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Retorno logaritmico barra a barra."""
    n = len(prices)
    out = np.zeros(n, dtype=np.float64)
    if n > 1:
        out[1:] = np.log((prices[1:] + 1e-10) / (prices[:-1] + 1e-10))
    return out


def rate_of_change(prices: np.ndarray, period: int) -> np.ndarray:
    """Variacao percentual do close em relacao a N barras atras."""
    n = len(prices)
    out = np.zeros(n, dtype=np.float64)
    lag = max(1, int(period))
    if n > lag:
        base = prices[:-lag]
        out[lag:] = (prices[lag:] - base) / (base + 1e-10)
    return out


def delta_series(values: np.ndarray) -> np.ndarray:
    """Primeira diferenca da serie com zero na barra inicial."""
    n = len(values)
    out = np.zeros(n, dtype=np.float64)
    if n > 1:
        out[1:] = np.diff(values)
    return out


def rolling_realized_vol_ratio(log_return: np.ndarray, target_vol: float, window: int) -> np.ndarray:
    """Desvio padrao rolling dos retornos log normalizado pela vol alvo do indice."""
    n = len(log_return)
    out = np.zeros(n, dtype=np.float64)
    span = max(2, int(window))
    scale = max(float(target_vol), 1e-10)
    for i in range(span, n):
        segment = log_return[max(0, i - span + 1) : i + 1]
        out[i] = float(np.std(segment)) / scale
    return out


def price_zscore(prices: np.ndarray, window: int) -> np.ndarray:
    """Z-score do close em relacao a media movel e desvio na janela."""
    n = len(prices)
    out = np.zeros(n, dtype=np.float64)
    w = max(2, int(window))
    for i in range(n):
        start = max(0, i - w + 1)
        segment = prices[start : i + 1]
        mean = float(np.mean(segment))
        std = float(np.std(segment))
        out[i] = (float(prices[i]) - mean) / (std + 1e-10)
    return out


def ema_distances(prices: np.ndarray, span_20: int, span_50: int) -> tuple[np.ndarray, np.ndarray]:
    """Distancia percentual do close para EMA20 e EMA50."""
    df = pl.DataFrame({"close": prices})
    ema_20 = df.select(pl.col("close").ewm_mean(span=int(span_20))).to_numpy().flatten()
    ema_50 = df.select(pl.col("close").ewm_mean(span=int(span_50))).to_numpy().flatten()
    dist_20 = (prices - ema_20) / (ema_20 + 1e-10)
    dist_50 = (prices - ema_50) / (ema_50 + 1e-10)
    return dist_20, dist_50


def calculate_macd(
    prices: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula a linha MACD e a linha de sinal (normalizadas pelo preco)."""
    df = pl.DataFrame({"close": prices})
    ema_fast = df.select(pl.col("close").ewm_mean(span=int(fast_period))).to_numpy().flatten()
    ema_slow = df.select(pl.col("close").ewm_mean(span=int(slow_period))).to_numpy().flatten()
    macd = ema_fast - ema_slow
    df_macd = pl.DataFrame({"macd": macd})
    macd_signal = df_macd.select(pl.col("macd").ewm_mean(span=int(signal_period))).to_numpy().flatten()
    return macd / (prices + 1e-10), macd_signal / (prices + 1e-10)


def calculate_stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
    smooth_k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula Stochastic Oscillator %K e %D normalizados (escala de 0.0 a 1.0)."""
    n = len(close)
    k_line = np.full(n, 0.5, dtype=np.float64)
    if n < period:
        return k_line, k_line
    for i in range(period - 1, n):
        start = i - period + 1
        h_val = np.max(high[start : i + 1])
        l_val = np.min(low[start : i + 1])
        denom = h_val - l_val
        if denom > 1e-10:
            k_line[i] = (close[i] - l_val) / denom
        else:
            k_line[i] = 0.5
    df_k = pl.DataFrame({"k": k_line})
    smooth_k_arr = df_k.select(pl.col("k").rolling_mean(window_size=smooth_k, min_samples=1)).to_numpy().flatten()
    d_arr = (
        pl.DataFrame({"sk": smooth_k_arr})
        .select(pl.col("sk").rolling_mean(window_size=smooth_k, min_samples=1))
        .to_numpy()
        .flatten()
    )
    return smooth_k_arr, d_arr


def calculate_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
    """Calcula Commodity Channel Index (CCI) normalizado (dividido por 100)."""
    n = len(close)
    tp = (high + low + close) / 3.0
    cci = np.zeros(n, dtype=np.float64)
    w = max(2, int(period))
    for i in range(n):
        start = max(0, i - w + 1)
        segment = tp[start : i + 1]
        ma = float(np.mean(segment))
        mad = float(np.mean(np.abs(segment - ma)))
        if mad > 1e-10:
            cci[i] = (tp[i] - ma) / (0.015 * mad)
        else:
            cci[i] = 0.0
    return cci / 100.0

"""Indicadores tecnicos normalizados para features DL Rise/Fall."""

import numpy as np
import polars as pl


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
        "rsi_period": 14,
        "roc_period": 10,
        "vol_window": 10,
        "ema_20": 20,
        "ema_50": 50,
        "bb_window": 20,
        "atr_window": 14,
        "hurst_window": 64,
        "vr_short": 2,
        "vr_long": 8,
        "rel_vol_span": 50,
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

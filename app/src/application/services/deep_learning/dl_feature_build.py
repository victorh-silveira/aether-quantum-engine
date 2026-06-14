"""Series de preco e tensores de features para classificacao Rise/Fall."""

import numpy as np
import polars as pl

from src.application.services.deep_learning.dl_hurst import hurst_exponent, variance_ratio


MICRO_FEATURE_DIM = 5
TRADITIONAL_FEATURE_DIM = 8
VOLATILITY_FEATURE_DIM = 3
PERSISTENCE_FEATURE_DIM = 2
FEATURE_DIM = MICRO_FEATURE_DIM + TRADITIONAL_FEATURE_DIM + VOLATILITY_FEATURE_DIM + PERSISTENCE_FEATURE_DIM


def symbol_vol_target(symbol: str) -> float:
    """Volatilidade anualizada alvo do indice sintetico Deriv (ex. R_75 -> 0.75)."""
    parts = str(symbol).split("_")
    if len(parts) >= 2:
        try:
            return float(parts[-1]) / 100.0
        except ValueError:
            pass
    return 0.50


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


def _feature_windows(granularity: int) -> dict[str, int]:
    """Janelas de indicadores para contratos alinhados a barras de 60s."""
    _ = max(1, int(granularity))
    return {
        "rsi_period": 14,
        "vol_window": 10,
        "ema_fast": 8,
        "ema_micro": 5,
        "ema_short": 13,
        "ema_slow": 21,
        "sma_window": 20,
        "bb_window": 20,
        "atr_window": 14,
        "hurst_window": 64,
        "vr_short": 2,
        "vr_long": 8,
        "rel_vol_span": 50,
    }


def _bollinger(prices: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _atr_norm(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
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


def _default_micro(n: int) -> dict[str, np.ndarray]:
    """Microestrutura neutra quando ticks ainda nao foram agregados."""
    return {
        "tick_count": np.zeros(n, dtype=np.float64),
        "mean_inter_tick_ms": np.zeros(n, dtype=np.float64),
        "price_velocity": np.zeros(n, dtype=np.float64),
        "price_acceleration": np.zeros(n, dtype=np.float64),
        "consecutive_diff_std": np.zeros(n, dtype=np.float64),
    }


def attach_microstructure(
    series: dict[str, np.ndarray],
    micro: dict[str, np.ndarray] | None,
) -> None:
    """Anexa arrays de microestrutura por barra ao dicionario de series."""
    n = len(series["returns"])
    if not micro:
        series.update(_default_micro(n))
        return
    for key in (
        "tick_count",
        "mean_inter_tick_ms",
        "price_velocity",
        "price_acceleration",
        "consecutive_diff_std",
    ):
        arr = micro.get(key)
        if arr is None or len(arr) != n:
            series[key] = _default_micro(n)[key]
        else:
            series[key] = np.asarray(arr[:n], dtype=np.float64)


def precompute_price_series(
    prices: np.ndarray,
    *,
    granularity: int = 60,
    symbol: str = "R_50",
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    micro: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Precomputa series auxiliares usadas na montagem de features."""
    win = _feature_windows(granularity)
    n = len(prices)
    returns = np.zeros(n, dtype=np.float64)
    if n > 1:
        returns[1:] = np.diff(prices) / (prices[:-1] + 1e-10)
    vol_window = int(win["vol_window"])
    vol = np.zeros(n, dtype=np.float64)
    for i in range(vol_window, n):
        vol[i] = np.std(returns[max(0, i - vol_window) : i])
    rsi = calculate_rsi(prices, period=int(win["rsi_period"])) / 100.0
    df = pl.DataFrame({"close": prices})
    ema_fast = df.select(pl.col("close").ewm_mean(span=int(win["ema_fast"]))).to_numpy().flatten()
    ema_slow = df.select(pl.col("close").ewm_mean(span=int(win["ema_slow"]))).to_numpy().flatten()
    ema_spread = (ema_fast - ema_slow) / (ema_slow + 1e-10)
    ema_micro = df.select(pl.col("close").ewm_mean(span=int(win["ema_micro"]))).to_numpy().flatten()
    ema_short = df.select(pl.col("close").ewm_mean(span=int(win["ema_short"]))).to_numpy().flatten()
    ema_micro_spread = (ema_micro - ema_short) / (ema_short + 1e-10)
    sma_window = int(win["sma_window"])
    sma20 = df.select(pl.col("close").rolling_mean(window_size=sma_window, min_samples=1)).to_numpy().flatten()
    sma_dist = (prices - sma20) / (sma20 + 1e-10)
    if high is not None and low is not None and len(high) >= n and len(low) >= n:
        h = np.asarray(high[:n], dtype=np.float64)
        low_px = np.asarray(low[:n], dtype=np.float64)
    elif open_ is not None and len(open_) >= n:
        o = np.asarray(open_[:n], dtype=np.float64)
        close_px = prices.astype(np.float64)
        h = np.maximum(close_px, o)
        low_px = np.minimum(close_px, o)
    else:
        h = prices.astype(np.float64)
        low_px = prices.astype(np.float64)
    bb_lower, bb_mid, bb_upper = _bollinger(prices, int(win["bb_window"]))
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    bb_pct_b = (prices - bb_lower) / (bb_upper - bb_lower + 1e-10)
    atr_norm = _atr_norm(h, low_px, prices.astype(np.float64), int(win["atr_window"]))
    target_vol = symbol_vol_target(symbol)
    vol_vs_target = vol / (target_vol + 1e-10)
    vol_z = np.zeros(n, dtype=np.float64)
    rel_span = int(win["rel_vol_span"])
    for i in range(vol_window, n):
        base = np.mean(vol[max(0, i - rel_span) : i + 1]) + 1e-10
        vol_z[i] = (vol[i] - base) / base
    hurst = hurst_exponent(prices, window=int(win["hurst_window"]))
    vr = variance_ratio(prices, short=int(win["vr_short"]), long=int(win["vr_long"]))
    series = {
        "returns": returns,
        "vol": vol,
        "rsi": rsi,
        "bb_pct_b": bb_pct_b,
        "bb_width": bb_width,
        "atr_norm": atr_norm,
        "ema_spread": ema_spread,
        "ema_micro_spread": ema_micro_spread,
        "sma_dist": sma_dist,
        "vol_vs_target": vol_vs_target,
        "vol_z": vol_z,
        "hurst": hurst,
        "variance_ratio": vr,
    }
    attach_microstructure(series, micro)
    for k, v in series.items():
        series[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return series


def build_feature_row(series: dict[str, np.ndarray], index: int) -> np.ndarray:
    """Monta vetor de features na barra indicada."""
    micro = np.array(
        [
            series["tick_count"][index],
            series["mean_inter_tick_ms"][index] / 1000.0,
            series["price_velocity"][index],
            series["price_acceleration"][index],
            series["consecutive_diff_std"][index],
        ],
        dtype=np.float32,
    )
    traditional = np.array(
        [
            series["rsi"][index],
            series["bb_pct_b"][index],
            series["bb_width"][index],
            series["atr_norm"][index],
            series["ema_spread"][index],
            series["ema_micro_spread"][index],
            series["sma_dist"][index],
            series["returns"][index - 1] if index > 0 else 0.0,
        ],
        dtype=np.float32,
    )
    volatility = np.array(
        [
            series["vol"][index],
            series["vol_vs_target"][index],
            series["vol_z"][index],
        ],
        dtype=np.float32,
    )
    persistence = np.array(
        [
            series["hurst"][index],
            series["variance_ratio"][index],
        ],
        dtype=np.float32,
    )
    return np.concatenate([micro, traditional, volatility, persistence], axis=0).astype(np.float32)


def build_feature_matrix(
    series: dict[str, np.ndarray],
) -> np.ndarray:
    """Monta matriz (N, FEATURE_DIM) com uma linha de features por barra."""
    n = len(series["returns"])
    rows = [build_feature_row(series, i) for i in range(n)]
    return np.stack(rows, axis=0).astype(np.float32)


def build_sequence_tensor(
    prices: np.ndarray,
    lookback: int,
    end_index: int,
    *,
    granularity: int = 60,
    symbol: str = "R_50",
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    micro: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Monta janela (lookback, FEATURE_DIM) terminando em end_index inclusive."""
    series = precompute_price_series(
        prices,
        granularity=granularity,
        symbol=symbol,
        open_=open_,
        high=high,
        low=low,
        micro=micro,
    )
    start = end_index - lookback + 1
    rows = [build_feature_row(series, i) for i in range(start, end_index + 1)]
    return np.stack(rows, axis=0).astype(np.float32)

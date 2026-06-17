"""Series de preco e tensores de features para classificacao Rise/Fall."""

import numpy as np

from src.application.services.deep_learning.dl_feature_indicators import (
    atr_norm,
    bollinger,
    calculate_adx,
    calculate_cci,
    calculate_cmo,
    calculate_ema_crossover,
    calculate_keltner_channel_pct_b,
    calculate_macd,
    calculate_rsi,
    calculate_stochastic,
    calculate_volatility_ratio,
    calculate_williams_r,
    delta_series,
    ema_distances,
    feature_windows,
    log_returns,
    price_zscore,
    rate_of_change,
    rolling_realized_vol_ratio,
)
from src.application.services.deep_learning.dl_hurst import hurst_exponent, variance_ratio


_feature_windows = feature_windows


MICRO_FEATURE_DIM = 5
TRADITIONAL_FEATURE_DIM = 22
VOLATILITY_FEATURE_DIM = 5
PERSISTENCE_FEATURE_DIM = 2
FEATURE_DIM = MICRO_FEATURE_DIM + TRADITIONAL_FEATURE_DIM + VOLATILITY_FEATURE_DIM + PERSISTENCE_FEATURE_DIM


def symbol_vol_target(symbol: str) -> float:
    """Volatilidade anualizada alvo do indice sintetico Deriv (ex. R_75 -> 0.75)."""
    parts = str(symbol).split("_")
    try:
        return float(parts[-1]) / 100.0 if len(parts) >= 2 else 0.50
    except ValueError:
        return 0.50


def _default_micro(n: int) -> dict[str, np.ndarray]:
    """Microestrutura neutra quando ticks ainda nao foram agregados."""
    keys = ("tick_count", "mean_inter_tick_ms", "price_velocity", "price_acceleration", "consecutive_diff_std")
    return {k: np.zeros(n, dtype=np.float64) for k in keys}


def attach_microstructure(
    series: dict[str, np.ndarray],
    micro: dict[str, np.ndarray] | None,
) -> None:
    """Anexa arrays de microestrutura por barra ao dicionario de series."""
    n = len(series["log_return"])
    if not micro:
        series.update(_default_micro(n))
        return
    defaults = _default_micro(n)
    for key in defaults:
        arr = micro.get(key)
        if arr is None or len(arr) != n:
            series[key] = defaults[key]
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
    implied_vol_bars: int = 60,
) -> dict[str, np.ndarray]:
    """Precomputa series auxiliares usadas na montagem de features."""
    win = feature_windows(granularity)
    n = len(prices)
    close = prices.astype(np.float64)
    log_return = log_returns(close)
    vol_window = int(win["vol_window"])
    vol = np.zeros(n, dtype=np.float64)
    for i in range(vol_window, n):
        vol[i] = np.std(log_return[max(0, i - vol_window) : i])
    rsi = calculate_rsi(prices, period=int(win["rsi_period"])) / 100.0
    delta_rsi = delta_series(rsi)
    ema_dist_20, ema_dist_50 = ema_distances(close, int(win["ema_20"]), int(win["ema_50"]))
    roc = rate_of_change(close, int(win["roc_period"]))
    if high is not None and low is not None and len(high) >= n and len(low) >= n:
        h = np.asarray(high[:n], dtype=np.float64)
        low_px = np.asarray(low[:n], dtype=np.float64)
    elif open_ is not None and len(open_) >= n:
        o = np.asarray(open_[:n], dtype=np.float64)
        h = np.maximum(close, o)
        low_px = np.minimum(close, o)
    else:
        h = close
        low_px = close
    bb_lower, bb_mid, bb_upper = bollinger(prices, int(win["bb_window"]))
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    bb_pct_b = (prices - bb_lower) / (bb_upper - bb_lower + 1e-10)
    atr = atr_norm(h, low_px, close, int(win["atr_window"]))
    target_vol = symbol_vol_target(symbol)
    vol_vs_target = vol / (target_vol + 1e-10)
    vol_z = np.zeros(n, dtype=np.float64)
    rel_span = int(win["rel_vol_span"])
    for i in range(vol_window, n):
        base = np.mean(vol[max(0, i - rel_span) : i + 1]) + 1e-10
        vol_z[i] = (vol[i] - base) / base
    hurst = hurst_exponent(prices, window=int(win["hurst_window"]))
    vr = variance_ratio(prices, short=int(win["vr_short"]), long=int(win["vr_long"]))
    zscore = price_zscore(close, int(win["bb_window"]))
    implied_vol = rolling_realized_vol_ratio(log_return, target_vol, implied_vol_bars)

    # Computar novos indicadores
    macd, macd_signal = calculate_macd(
        close,
        fast_period=int(win["macd_fast"]),
        slow_period=int(win["macd_slow"]),
        signal_period=int(win["macd_signal"]),
    )
    stoch_k, stoch_d = calculate_stochastic(
        h,
        low_px,
        close,
        period=int(win["stoch_period"]),
        smooth_k=int(win["stoch_smooth"]),
    )
    cci = calculate_cci(h, low_px, close, period=int(win["cci_period"]))
    adx, di_diff = calculate_adx(h, low_px, close, period=int(win["adx_period"]))
    williams_r = calculate_williams_r(h, low_px, close, period=int(win["williams_period"]))
    ema_9_21_dist = calculate_ema_crossover(
        close,
        fast=int(win["ema_fast_crossover"]),
        slow=int(win["ema_slow_crossover"]),
    )
    roc_rsi = rate_of_change(rsi, period=int(win["roc_period"]))
    vol_ratio_short_long = calculate_volatility_ratio(
        log_return,
        short=int(win["vol_ratio_short"]),
        long=int(win["vol_ratio_long"]),
    )
    cmo = calculate_cmo(close, period=int(win["cmo_period"]))
    keltner_pct_b = calculate_keltner_channel_pct_b(
        h,
        low_px,
        close,
        period=int(win["kc_period"]),
        atr_period=int(win["kc_atr_period"]),
    )

    series = {
        "adx": adx,
        "atr_norm": atr,
        "bb_pct_b": bb_pct_b,
        "bb_width": bb_width,
        "cci": cci,
        "delta_rsi": delta_rsi,
        "di_diff": di_diff,
        "ema_9_21_dist": ema_9_21_dist,
        "ema_dist_20": ema_dist_20,
        "ema_dist_50": ema_dist_50,
        "hurst": hurst,
        "implied_vol_ratio": implied_vol,
        "log_return": log_return,
        "macd": macd,
        "macd_signal": macd_signal,
        "price_zscore": zscore,
        "roc": roc,
        "roc_rsi": roc_rsi,
        "rsi": rsi,
        "stoch_d": stoch_d,
        "stoch_k": stoch_k,
        "variance_ratio": vr,
        "vol": vol,
        "vol_ratio_short_long": vol_ratio_short_long,
        "vol_vs_target": vol_vs_target,
        "vol_z": vol_z,
        "williams_r": williams_r,
        "cmo": cmo,
        "keltner_pct_b": keltner_pct_b,
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
            series["delta_rsi"][index],
            series["bb_pct_b"][index],
            series["bb_width"][index],
            series["atr_norm"][index],
            series["ema_dist_20"][index],
            series["ema_dist_50"][index],
            series["log_return"][index],
            series["roc"][index],
            series["price_zscore"][index],
            series["macd"][index],
            series["macd_signal"][index],
            series["stoch_k"][index],
            series["stoch_d"][index],
            series["cci"][index],
            series["adx"][index],
            series["di_diff"][index],
            series["williams_r"][index],
            series["ema_9_21_dist"][index],
            series["roc_rsi"][index],
            series["cmo"][index],
            series["keltner_pct_b"][index],
        ],
        dtype=np.float32,
    )
    volatility = np.array(
        [
            series["vol"][index],
            series["vol_vs_target"][index],
            series["vol_z"][index],
            series["implied_vol_ratio"][index],
            series["vol_ratio_short_long"][index],
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
    n = len(series["log_return"])
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
    implied_vol_bars: int = 60,
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
        implied_vol_bars=implied_vol_bars,
    )
    start = end_index - lookback + 1
    rows = [build_feature_row(series, i) for i in range(start, end_index + 1)]
    return np.stack(rows, axis=0).astype(np.float32)

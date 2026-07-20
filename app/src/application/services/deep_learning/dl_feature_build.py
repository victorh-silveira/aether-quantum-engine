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
    """Volatilidade anualizada alvo do indice sintetico Deriv."""
    parts = str(symbol).split("_")
    try:
        return float(parts[-1]) / 100.0 if len(parts) >= 2 else 0.50
    except ValueError:
        return 0.50


def rolling_zscore_1024_fast(series: np.ndarray) -> np.ndarray:
    """Calcula Z-Score adaptativo historico com limite de 1024 periodos e clipping de +-3.0."""
    n = len(series)
    if n == 0:
        return np.zeros(0, dtype=np.float64)  # pragma: no cover
    means = np.zeros(n, dtype=np.float64)
    stds = np.zeros(n, dtype=np.float64)
    cumsum = np.cumsum(series)
    cumsum = np.insert(cumsum, 0, 0.0)
    cumsum2 = np.cumsum(series**2)
    cumsum2 = np.insert(cumsum2, 0, 0.0)
    for i in range(n):
        start = max(0, i - 1023)
        count = i - start + 1
        window_sum = cumsum[i + 1] - cumsum[start]
        window_sum2 = cumsum2[i + 1] - cumsum2[start]
        mean = window_sum / count
        var = max(0.0, (window_sum2 / count) - (mean**2))
        std = np.sqrt(var)
        means[i] = mean
        stds[i] = std
    z = (series - means) / (stds + 1e-12)
    return np.clip(z, -3.0, 3.0)


def _default_micro(n: int) -> dict[str, np.ndarray]:
    """Microestrutura neutra quando ticks ainda nao foram agregados."""
    keys = (
        "tick_count",
        "mean_inter_tick_ms",
        "price_velocity",
        "price_acceleration",
        "consecutive_diff_std",
        "micro_bid_ask_spread_momentum",
        "volatility_shadow_ratio",
    )
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


def compute_oscillators(
    h: np.ndarray,
    low_px: np.ndarray,
    close: np.ndarray,
    rsi: np.ndarray,
    log_return: np.ndarray,
    win: dict[str, float],
) -> dict[str, np.ndarray]:
    """Precomputa osciladores e cruzamentos auxiliares."""
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
    cmo = calculate_cmo(close, period=int(win["cmo_period"]))
    roc_rsi = rate_of_change(rsi, period=int(win["roc_period"]))
    ema_9_21_dist = calculate_ema_crossover(
        close,
        fast=int(win["ema_fast_crossover"]),
        slow=int(win["ema_slow_crossover"]),
    )
    vol_ratio_short_long = calculate_volatility_ratio(
        log_return,
        short=int(win["vol_ratio_short"]),
        long=int(win["vol_ratio_long"]),
    )
    keltner_pct_b = calculate_keltner_channel_pct_b(
        h,
        low_px,
        close,
        period=int(win["kc_period"]),
        atr_period=int(win["kc_atr_period"]),
    )
    return {
        "macd": macd,
        "macd_signal": macd_signal,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "cci": cci,
        "adx": adx,
        "di_diff": di_diff,
        "williams_r": williams_r,
        "cmo": cmo,
        "roc_rsi": roc_rsi,
        "ema_9_21_dist": ema_9_21_dist,
        "vol_ratio_short_long": vol_ratio_short_long,
        "keltner_pct_b": keltner_pct_b,
    }


def _resolve_high_low(
    close: np.ndarray,
    n: int,
    open_: np.ndarray | None,
    high: np.ndarray | None,
    low: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve arrays high/low a partir de OHLC disponivel."""
    if high is not None and low is not None and len(high) >= n and len(low) >= n:
        return np.asarray(high[:n], dtype=np.float64), np.asarray(low[:n], dtype=np.float64)
    if open_ is not None and len(open_) >= n:
        o = np.asarray(open_[:n], dtype=np.float64)
        return np.maximum(close, o), np.minimum(close, o)
    return close, close


def _rolling_vol_and_z(
    log_return: np.ndarray,
    n: int,
    vol_window: int,
    rel_vol_span: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula volatilidade rolling e z-score relativo."""
    vol = np.zeros(n, dtype=np.float64)
    for i in range(vol_window, n):
        vol[i] = np.std(log_return[max(0, i - vol_window) : i])
    vol_z = np.zeros(n, dtype=np.float64)
    for i in range(vol_window, n):
        base = np.mean(vol[max(0, i - rel_vol_span) : i + 1]) + 1e-10
        vol_z[i] = (vol[i] - base) / base
    return vol, vol_z


def _attach_micro_zscores(series: dict[str, np.ndarray], n: int) -> None:
    """Anexa micro features e z-scores adaptativos com clipping."""
    micro_mom = series.get("micro_bid_ask_spread_momentum", np.zeros(n, dtype=np.float64))
    shadow_ratio = series.get("volatility_shadow_ratio", np.zeros(n, dtype=np.float64))
    series["micro_bid_ask_spread_momentum"] = micro_mom
    series["micro_bid_ask_spread_momentum_zscore"] = rolling_zscore_1024_fast(micro_mom)
    series["volatility_shadow_ratio"] = shadow_ratio
    series["volatility_shadow_ratio_zscore"] = rolling_zscore_1024_fast(shadow_ratio)


def precompute_price_series(
    prices: np.ndarray,
    *,
    granularity: int = 60,
    symbol: str = "R_10",
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
    vol, vol_z = _rolling_vol_and_z(log_return, n, vol_window, int(win["rel_vol_span"]))
    rsi = calculate_rsi(prices, period=int(win["rsi_period"])) / 100.0
    delta_rsi = delta_series(rsi)
    ema_dist_20, ema_dist_50 = ema_distances(close, int(win["ema_20"]), int(win["ema_50"]))
    roc = rate_of_change(close, int(win["roc_period"]))
    h, low_px = _resolve_high_low(close, n, open_, high, low)
    bb_lower, bb_mid, bb_upper = bollinger(prices, int(win["bb_window"]))
    bb_w_raw = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    bb_width = np.clip((bb_w_raw - np.mean(bb_w_raw)) / (np.std(bb_w_raw) + 1e-10), -3.0, 3.0)
    bb_pct_b = (prices - bb_lower) / (bb_upper - bb_lower + 1e-10)
    atr_raw = atr_norm(h, low_px, close, int(win["atr_window"]))
    atr = np.clip((atr_raw - np.mean(atr_raw)) / (np.std(atr_raw) + 1e-10), -3.0, 3.0)
    target_vol = symbol_vol_target(symbol)
    vol_vs_target = vol / (target_vol + 1e-10)
    hurst = hurst_exponent(prices, window=int(win["hurst_window"]))
    vr = variance_ratio(prices, short=int(win["vr_short"]), long=int(win["vr_long"]))
    zscore = price_zscore(close, int(win["bb_window"]))
    implied_vol = rolling_realized_vol_ratio(log_return, target_vol, implied_vol_bars)
    osc = compute_oscillators(h, low_px, close, rsi, log_return, win)
    series = {
        "adx": osc["adx"],
        "atr_norm": atr,
        "bb_pct_b": bb_pct_b,
        "bb_width": bb_width,
        "cci": osc["cci"],
        "delta_rsi": delta_rsi,
        "di_diff": osc["di_diff"],
        "ema_9_21_dist": osc["ema_9_21_dist"],
        "ema_dist_20": ema_dist_20,
        "ema_dist_50": ema_dist_50,
        "hurst": hurst,
        "implied_vol_ratio": implied_vol,
        "log_return": log_return,
        "macd": osc["macd"],
        "macd_signal": osc["macd_signal"],
        "price_zscore": zscore,
        "roc": roc,
        "roc_rsi": osc["roc_rsi"],
        "rsi": rsi,
        "stoch_d": osc["stoch_d"],
        "stoch_k": osc["stoch_k"],
        "variance_ratio": vr,
        "vol": vol,
        "vol_ratio_short_long": osc["vol_ratio_short_long"],
        "vol_vs_target": vol_vs_target,
        "vol_z": vol_z,
        "williams_r": osc["williams_r"],
        "cmo": osc["cmo"],
        "keltner_pct_b": osc["keltner_pct_b"],
    }
    attach_microstructure(series, micro)
    _attach_micro_zscores(series, n)
    for k, v in series.items():
        series[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return series

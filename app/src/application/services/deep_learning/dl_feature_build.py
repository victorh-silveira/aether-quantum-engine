"""Series de preco e tensores de features para classificacao Rise/Fall."""

import numpy as np

from src.application.services.deep_learning.dl_feature_indicators import (
    atr_norm,
    bollinger,
    calculate_rsi,
    delta_series,
    ema_distances,
    feature_windows,
    log_returns,
    price_zscore,
    rate_of_change,
    rolling_realized_vol_ratio,
)
from src.application.services.deep_learning.dl_feature_oscillators import (
    compute_oscillators,
    rolling_zscore_fast,
)
from src.application.services.deep_learning.dl_hurst import hurst_exponent, variance_ratio
from src.application.services.deep_learning.dl_indicator_config import load_indicator_config_from_settings


_feature_windows = feature_windows


MICRO_FEATURE_DIM = 5
TRADITIONAL_FEATURE_DIM = 22
VOLATILITY_FEATURE_DIM = 5
PERSISTENCE_FEATURE_DIM = 2
FEATURE_DIM = MICRO_FEATURE_DIM + TRADITIONAL_FEATURE_DIM + VOLATILITY_FEATURE_DIM + PERSISTENCE_FEATURE_DIM


def symbol_vol_target(symbol: str) -> float:
    """Volatilidade anualizada alvo por simbolo (Volatility R_* ou legado)."""
    key = str(symbol).upper()
    if key == "R_10":
        return 0.16
    parts = key.split("_")
    try:
        return float(parts[-1]) / 100.0 if len(parts) >= 2 else 0.50
    except ValueError:
        return 0.50


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


def _attach_micro_zscores(
    series: dict[str, np.ndarray],
    n: int,
    *,
    zscore_window: int,
    zscore_clip: float,
) -> None:
    """Anexa micro features e z-scores adaptativos com clipping."""
    micro_mom = series.get("micro_bid_ask_spread_momentum", np.zeros(n, dtype=np.float64))
    shadow_ratio = series.get("volatility_shadow_ratio", np.zeros(n, dtype=np.float64))
    series["micro_bid_ask_spread_momentum"] = micro_mom
    series["micro_bid_ask_spread_momentum_zscore"] = rolling_zscore_fast(
        micro_mom, window=zscore_window, clip=zscore_clip
    )
    series["volatility_shadow_ratio"] = shadow_ratio
    series["volatility_shadow_ratio_zscore"] = rolling_zscore_fast(shadow_ratio, window=zscore_window, clip=zscore_clip)


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
    indicator_cfg: dict | None = None,
) -> dict[str, np.ndarray]:
    """Precomputa series auxiliares usadas na montagem de features."""
    cfg = indicator_cfg if isinstance(indicator_cfg, dict) else load_indicator_config_from_settings()
    win = feature_windows(granularity, cfg["windows"])
    mult = cfg["multipliers"]
    norm = cfg["normalization"]
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
    bb_lower, bb_mid, bb_upper = bollinger(prices, int(win["bb_window"]), std_mult=float(mult["bb_std_mult"]))
    bb_w_raw = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    bb_clip = float(norm["bb_width_z_clip"])
    bb_width = np.clip((bb_w_raw - np.mean(bb_w_raw)) / (np.std(bb_w_raw) + 1e-10), -bb_clip, bb_clip)
    bb_pct_b = (prices - bb_lower) / (bb_upper - bb_lower + 1e-10)
    atr_raw = atr_norm(h, low_px, close, int(win["atr_window"]))
    atr_clip = float(norm["atr_z_clip"])
    atr = np.clip((atr_raw - np.mean(atr_raw)) / (np.std(atr_raw) + 1e-10), -atr_clip, atr_clip)
    target_vol = symbol_vol_target(symbol)
    vol_vs_target = vol / (target_vol + 1e-10)
    hurst = hurst_exponent(
        prices,
        window=int(win["hurst_window"]),
        min_window=int(win["hurst_min_window"]),
    )
    vr = variance_ratio(prices, short=int(win["vr_short"]), long=int(win["vr_long"]))
    zscore = price_zscore(close, int(win["bb_window"]))
    implied_vol = rolling_realized_vol_ratio(log_return, target_vol, implied_vol_bars)
    osc = compute_oscillators(
        h,
        low_px,
        close,
        rsi,
        log_return,
        win,
        cci_constant=float(mult["cci_constant"]),
        kc_atr_mult=float(mult["kc_atr_mult"]),
    )
    series = {
        "open": open_ if open_ is not None else close,
        "close": close,
        "high": h,
        "low": low_px,
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
    _attach_micro_zscores(
        series,
        n,
        zscore_window=int(win["zscore_micro_window"]),
        zscore_clip=float(norm["series_z_clip"]),
    )
    for k, v in series.items():
        series[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return series

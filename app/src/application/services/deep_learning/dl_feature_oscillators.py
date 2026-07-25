"""Osciladores e z-score micro para a matriz de features DL."""

from __future__ import annotations

import numpy as np

from src.application.services.deep_learning.dl_feature_indicators import (
    calculate_adx,
    calculate_cci,
    calculate_choppiness_index,
    calculate_cmo,
    calculate_ema_crossover,
    calculate_keltner_channel_pct_b,
    calculate_macd,
    calculate_stochastic,
    calculate_supertrend,
    calculate_volatility_ratio,
    calculate_vwap_zscore,
    calculate_williams_r,
    rate_of_change,
)
from src.application.services.deep_learning.dl_indicator_config import load_indicator_config_from_settings


def rolling_zscore_fast(series: np.ndarray, *, window: int, clip: float) -> np.ndarray:
    """Calcula Z-Score adaptativo historico com janela e clipping configuraveis."""
    n = len(series)
    if n == 0:
        return np.zeros(0, dtype=np.float64)  # pragma: no cover
    means = np.zeros(n, dtype=np.float64)
    stds = np.zeros(n, dtype=np.float64)
    cumsum = np.cumsum(series)
    cumsum = np.insert(cumsum, 0, 0.0)
    cumsum2 = np.cumsum(series**2)
    cumsum2 = np.insert(cumsum2, 0, 0.0)
    span = max(1, int(window))
    for i in range(n):
        start = max(0, i - span + 1)
        count = i - start + 1
        window_sum = cumsum[i + 1] - cumsum[start]
        window_sum2 = cumsum2[i + 1] - cumsum2[start]
        mean = window_sum / count
        var = max(0.0, (window_sum2 / count) - (mean**2))
        std = np.sqrt(var)
        means[i] = mean
        stds[i] = std
    z = (series - means) / (stds + 1e-12)
    bound = float(clip)
    return np.clip(z, -bound, bound)


def rolling_zscore_1024_fast(series: np.ndarray) -> np.ndarray:
    """Z-Score com janela/clip de deep_learning.indicators."""
    cfg = load_indicator_config_from_settings()
    return rolling_zscore_fast(
        series,
        window=int(cfg["windows"]["zscore_micro_window"]),
        clip=float(cfg["normalization"]["series_z_clip"]),
    )


def compute_oscillators(
    h: np.ndarray,
    low_px: np.ndarray,
    close: np.ndarray,
    rsi: np.ndarray,
    log_return: np.ndarray,
    win: dict[str, float],
    *,
    cci_constant: float,
    kc_atr_mult: float,
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
    cci = calculate_cci(h, low_px, close, period=int(win["cci_period"]), cci_constant=cci_constant)
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
        atr_mult=kc_atr_mult,
    )
    choppiness_index = calculate_choppiness_index(h, low_px, close, period=14)
    vwap_zscore = calculate_vwap_zscore(h, low_px, close, window=20)
    _, supertrend_dir = calculate_supertrend(h, low_px, close, period=10, multiplier=3.0)
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
        "choppiness_index": choppiness_index,
        "vwap_zscore": vwap_zscore,
        "supertrend_dir": supertrend_dir,
    }

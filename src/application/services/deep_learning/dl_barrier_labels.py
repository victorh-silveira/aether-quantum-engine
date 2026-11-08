"""Metodos de rotulagem por barreira tripla (Marcos Lopez de Prado) e quantica sem lookahead leakage."""

from __future__ import annotations

import numpy as np

from src.application.services.deep_learning.dl_regime_filter import (
    market_disequilibrium_polarity,
)


def triple_barrier_label_and_mask(
    prices: np.ndarray,
    index: int,
    horizon_bars: int,
    *,
    lookback_vol: int = 20,
    barrier_mult: float = 1.0,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    series: dict[str, np.ndarray] | None = None,
    dead_zone_ratio: float = 0.35,
) -> tuple[bool, float]:
    """Avalia rotulagem causal por barreira tripla sem lookahead leakage."""
    start_vol = max(0, index - lookback_vol)
    vol_seg = prices[start_vol : index + 1]
    if series is not None and "atr_raw" in series and index < len(series["atr_raw"]):
        sigma = float(series["atr_raw"][index])
    elif len(vol_seg) > 1:
        log_rets = np.diff(np.log(np.maximum(vol_seg, 1e-8)))
        sigma = float(np.std(log_rets)) if len(log_rets) > 0 else 0.001
    else:
        sigma = 0.001
    dyn_barrier = max(0.0003, sigma * barrier_mult) * float(prices[index])
    upper_barrier = prices[index] + dyn_barrier
    lower_barrier = prices[index] - dyn_barrier

    up: bool | None = None
    hit = False
    max_check = min(len(prices), index + max(1, int(horizon_bars)) + 1)
    for step_idx in range(index + 1, max_check):
        h_val = float(high[step_idx]) if high is not None and step_idx < len(high) else float(prices[step_idx])
        l_val = float(low[step_idx]) if low is not None and step_idx < len(low) else float(prices[step_idx])
        if h_val >= upper_barrier and l_val > lower_barrier:
            up, hit = True, True
            break
        if l_val <= lower_barrier and h_val < upper_barrier:
            up, hit = False, True
            break
        if prices[step_idx] >= upper_barrier and prices[step_idx] > lower_barrier:
            up, hit = True, True
            break
        if prices[step_idx] <= lower_barrier and prices[step_idx] < upper_barrier:
            up, hit = False, True
            break

    final_idx = min(len(prices) - 1, index + max(1, int(horizon_bars)))
    final_p = float(prices[final_idx])
    base_p = float(open_[final_idx]) if open_ is not None and final_idx < len(open_) else float(prices[index])
    delta = final_p - base_p

    if not hit:
        up = delta > 0.0

    target = bool(up)
    if series is not None:
        polarity = market_disequilibrium_polarity(series, index)
        mask = 1.0 if polarity != 0 else 0.0
    else:
        dead = dyn_barrier * float(dead_zone_ratio)
        mask = 1.0 if abs(delta) >= dead else 0.0

    return target, float(mask)


def triple_barrier_direction(
    prices: np.ndarray,
    index: int,
    horizon_bars: int,
    *,
    lookback_vol: int = 20,
    barrier_mult: float = 1.0,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    series: dict[str, np.ndarray] | None = None,
    dead_zone_ratio: float = 0.35,
) -> bool:
    """Avalia direcao do primeiro toque entre barreira superior e inferior."""
    up, _ = triple_barrier_label_and_mask(
        prices,
        index,
        horizon_bars,
        lookback_vol=lookback_vol,
        barrier_mult=barrier_mult,
        open_=open_,
        high=high,
        low=low,
        series=series,
        dead_zone_ratio=dead_zone_ratio,
    )
    return up


def quantum_multi_barrier_label_and_mask(
    prices: np.ndarray,
    index: int,
    horizon_bars: int,
    *,
    lookback_vol: int = 20,
    barrier_mult: float = 1.0,
    asymmetry_factor: float = 0.20,
    min_viable_delta: float | None = None,
    dead_zone_ratio: float = 0.35,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    series: dict[str, np.ndarray] | None = None,
) -> tuple[bool, float]:
    """Quantum Multi-Barrier com mascara causal baseada estritamente nas series em t."""
    start_vol = max(0, index - lookback_vol)
    vol_seg = prices[start_vol : index + 1]
    if series is not None and "atr_raw" in series and index < len(series["atr_raw"]):
        sigma = float(series["atr_raw"][index])
    elif len(vol_seg) > 1:
        log_rets = np.diff(np.log(np.maximum(vol_seg, 1e-8)))
        sigma = float(np.std(log_rets)) if len(log_rets) > 0 else 0.001
    else:
        sigma = 0.001

    trend_slope = float(prices[index] - prices[start_vol]) / float(max(1, index - start_vol))
    is_uptrend = trend_slope >= 0.0
    upper_mult = barrier_mult * (1.0 - asymmetry_factor if is_uptrend else 1.0 + asymmetry_factor)
    lower_mult = barrier_mult * (1.0 + asymmetry_factor if is_uptrend else 1.0 - asymmetry_factor)

    dyn_upper = max(0.0003, sigma * upper_mult) * float(prices[index])
    dyn_lower = max(0.0003, sigma * lower_mult) * float(prices[index])
    upper_barrier = prices[index] + dyn_upper
    lower_barrier = prices[index] - dyn_lower

    up: bool | None = None
    hit = False
    max_check = min(len(prices), index + max(1, int(horizon_bars)) + 1)
    for step_idx in range(index + 1, max_check):
        h_val = float(high[step_idx]) if high is not None and step_idx < len(high) else float(prices[step_idx])
        l_val = float(low[step_idx]) if low is not None and step_idx < len(low) else float(prices[step_idx])
        if h_val >= upper_barrier and l_val > lower_barrier:
            up, hit = True, True
            break
        if l_val <= lower_barrier and h_val < upper_barrier:
            up, hit = False, True
            break
        if prices[step_idx] >= upper_barrier and prices[step_idx] > lower_barrier:
            up, hit = True, True
            break
        if prices[step_idx] <= lower_barrier and prices[step_idx] < upper_barrier:
            up, hit = False, True
            break

    final_idx = min(len(prices) - 1, index + max(1, int(horizon_bars)))
    final_p = float(prices[final_idx])
    base_p = float(open_[final_idx]) if open_ is not None and final_idx < len(open_) else float(prices[index])
    delta = final_p - base_p

    if not hit:
        up = delta > 0.0 if abs(delta) > 1e-9 else is_uptrend

    target = bool(up)
    if series is not None:
        polarity = market_disequilibrium_polarity(series, index)
        mask = 1.0 if polarity != 0 else 0.0
    elif min_viable_delta is not None:
        threshold = float(min_viable_delta) * float(prices[index])
        mask = 1.0 if abs(delta) >= threshold else 0.0
    else:
        dead_zone = min(dyn_upper, dyn_lower) * float(dead_zone_ratio)
        mask = 1.0 if abs(delta) >= dead_zone else 0.0

    return target, float(mask)


def quantum_multi_barrier_direction(
    prices: np.ndarray,
    index: int,
    horizon_bars: int,
    *,
    lookback_vol: int = 20,
    barrier_mult: float = 1.0,
    asymmetry_factor: float = 0.20,
    min_viable_delta: float | None = None,
    dead_zone_ratio: float = 0.35,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    series: dict[str, np.ndarray] | None = None,
) -> bool:
    """Quantum Multi-Barrier: direcao causal sem lookahead leakage."""
    up, _ = quantum_multi_barrier_label_and_mask(
        prices,
        index,
        horizon_bars,
        lookback_vol=lookback_vol,
        barrier_mult=barrier_mult,
        asymmetry_factor=asymmetry_factor,
        min_viable_delta=min_viable_delta,
        dead_zone_ratio=dead_zone_ratio,
        open_=open_,
        high=high,
        low=low,
        series=series,
    )
    return up

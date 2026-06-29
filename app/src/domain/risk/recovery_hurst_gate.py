"""Piso logaritmico de recovery por Hurst e filtro de persistencia do pool."""

from __future__ import annotations

import math


def recovery_hurst_adjusted_floor(
    base_floor: float,
    hurst: float,
    *,
    consecutive_losses: int,
    hurst_persistence_min: float = 0.58,
    log_scale: float = 0.08,
) -> float:
    """Eleva piso de sinal quando Hurst nao indica persistencia em martingale N2+."""
    if int(consecutive_losses) < 2:
        return float(base_floor)
    h = float(hurst)
    target = float(hurst_persistence_min)
    if h + 1e-9 >= target:
        return float(base_floor)
    deficit = max(0.0, target - h)
    return float(base_floor) + float(log_scale) * math.log1p(deficit)


def recovery_pool_has_persistence(
    candidates: list[tuple],
    *,
    consecutive_losses: int,
    hurst_min: float = 0.58,
) -> bool:
    """True quando ao menos um candidato tem Hurst acima do minimo de persistencia."""
    if int(consecutive_losses) < 2:
        return True
    threshold = float(hurst_min)
    for item in candidates:
        if not isinstance(item, tuple) or len(item) < 3:
            continue
        metrics = item[2]
        if not isinstance(metrics, dict):
            continue
        indicators = metrics.get("indicators") or {}
        if float(indicators.get("hurst", 0.0)) + 1e-9 > threshold:
            return True
    return False

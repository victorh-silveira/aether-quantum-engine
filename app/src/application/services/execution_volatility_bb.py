"""Bollinger width efetivo com escala de volatilidade implicita."""

from __future__ import annotations

from src.application.services.deep_learning.dl_feature_build import symbol_vol_target


def _percentile_p10(values: list[float]) -> float:
    """Percentil 10 de uma lista numerica."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    index = max(0, int(0.10 * (len(ordered) - 1)))
    return ordered[index]


def bb_effective_width(
    *,
    bb_width: float,
    implied_vol_ratio: float,
    symbol: str,
    scale_enabled: bool = True,
) -> float:
    """Largura BB ajustada pela razao de vol implicita versus alvo do simbolo."""
    width = max(0.0, float(bb_width))
    if not scale_enabled:
        return width
    target = max(1e-6, symbol_vol_target(symbol))
    ratio = max(0.0, float(implied_vol_ratio)) / target
    return width * ratio


def squeeze_extreme_regime(
    *,
    bb_effective: float,
    bb_width_history: list[float] | None,
    vol_ratio: float,
    implied_vol_ratio: float,
    symbol: str,
    scale_enabled: bool = True,
) -> tuple[bool, float]:
    """True quando compressao extrema: BB efetivo abaixo do p10 e vol_ratio < 0.9."""
    hist = bb_width_history or [bb_effective]
    scaled_hist = [
        bb_effective_width(
            bb_width=v,
            implied_vol_ratio=implied_vol_ratio,
            symbol=symbol,
            scale_enabled=scale_enabled,
        )
        for v in hist
    ]
    p10 = _percentile_p10(scaled_hist)
    effective = bb_effective_width(
        bb_width=bb_effective,
        implied_vol_ratio=implied_vol_ratio,
        symbol=symbol,
        scale_enabled=scale_enabled,
    )
    extreme = effective < p10 and float(vol_ratio) < 0.9
    norm = 0.0 if p10 <= 0 else min(1.0, effective / max(p10, 1e-9))
    return extreme, norm


def squeeze_dynamic_min_edge(
    *,
    base_edge: float,
    bb_norm: float,
    squeeze_slope: float,
) -> float:
    """Sobe edge linearmente em squeeze: base + slope * (1 - bb_norm)."""
    slope = max(0.0, float(squeeze_slope))
    norm = max(0.0, min(1.0, float(bb_norm)))
    return float(base_edge) + slope * (1.0 - norm)

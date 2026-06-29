"""Penalizacao de peso DL por entropia adaptativa ao regime de volatilidade."""

from __future__ import annotations

from src.application.services.execution_volatility_threshold import volatility_regime_score
from src.domain.math.probability_entropy import (
    adaptive_entropy_ceiling,
    binary_entropy,
    entropy_penalty_factor,
)


def _regime_score_from_metrics(metrics: dict, dynamic_cfg: dict | None) -> float:
    """Obtem score de regime das metricas ou recalcula a partir de indicadores."""
    stored = metrics.get("volatility_regime")
    if stored is not None:
        return max(0.0, min(1.0, float(stored)))
    indicators = metrics.get("indicators") or {}
    if not indicators:
        return 0.0
    chunk = dynamic_cfg if isinstance(dynamic_cfg, dict) else {}
    return volatility_regime_score(
        bb_width=float(indicators.get("bb_width", 0.0)),
        atr_norm=float(indicators.get("atr_norm", 0.0)),
        adx=float(indicators.get("adx", 0.0)),
        vol_ratio=float(indicators.get("vol_ratio", 1.0)),
        bb_width_history=indicators.get("bb_width_history"),
        atr_norm_history=indicators.get("atr_norm_history"),
        cfg=chunk,
    )


def resolve_dl_entropy_penalty(
    probability: float,
    metrics: dict,
    *,
    calibration_cfg: dict | None = None,
    dynamic_cfg: dict | None = None,
) -> tuple[float, float, float]:
    """Retorna penalty, entropia e teto efetivo para peso DL."""
    cal = calibration_cfg if isinstance(calibration_cfg, dict) else {}
    base_ceiling = float(cal.get("entropy_ceiling", 0.92))
    floor = float(cal.get("entropy_floor", 0.0))
    tighten = float(cal.get("entropy_regime_tighten", 0.35))
    regime = _regime_score_from_metrics(metrics, dynamic_cfg)
    ceiling_eff = adaptive_entropy_ceiling(
        base_ceiling,
        regime,
        squeeze_tighten=tighten,
        entropy_floor=floor,
    )
    ent = binary_entropy(probability)
    penalty = entropy_penalty_factor(probability, ceiling=ceiling_eff, floor=floor)
    strength = float(cal.get("entropy_penalty_strength", 1.0))
    return min(1.0, penalty * max(0.0, strength)), ent, ceiling_eff

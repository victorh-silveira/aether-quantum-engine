"""Veto de inversao direcional em regime de expansao de volatilidade."""

from __future__ import annotations

from src.domain.models.trade import TradeDirection


_REVERSIVE_HINTS = frozenset({"exhaustion_flip", "mean_reversion"})


def apply_expansion_inversion_veto(
    exec_dir: TradeDirection,
    dl_dir: TradeDirection,
    hints: list[str],
    metrics: dict,
    *,
    exec_cfg: dict,
    clamp01,
) -> tuple[TradeDirection, list[str]]:
    """Veta inversao reversiva quando vol_ratio indica breakout/expansao."""
    if exec_dir == dl_dir:
        return exec_dir, hints
    if not _REVERSIVE_HINTS.intersection(hints):
        return exec_dir, hints
    indicators = metrics.get("indicators") or {}
    vol_ratio = float(indicators.get("vol_ratio", 1.0))
    threshold = float(exec_cfg.get("expansion_inversion_veto_vol_ratio", 1.15))
    if vol_ratio <= threshold:
        return exec_dir, hints
    retention = float(exec_cfg.get("expansion_inversion_score_retention", 0.70))
    momentum_scale = float(exec_cfg.get("expansion_momentum_kelly_scale", 0.85))
    prev_kelly = float(metrics.get("kelly_fraction_scale", 1.0))
    metrics["kelly_fraction_scale"] = prev_kelly * momentum_scale
    metrics["expansion_momentum_smoothing"] = momentum_scale
    metrics["expansion_inversion_veto"] = True
    metrics["expansion_inversion_score_retention"] = retention
    chosen = max(float(metrics.get("direction_call_score", 0.0)), float(metrics.get("direction_put_score", 0.0)))
    side_strength = clamp01(chosen) * retention
    metrics["trade_score"] = side_strength
    metrics["resolved_conviction"] = side_strength
    metrics["exec_direction"] = dl_dir.name
    metrics["resolved_direction"] = dl_dir.name
    metrics["direction_inverted"] = False
    if "expansion_veto" not in hints:
        hints = [*hints, "expansion_veto"]
    return dl_dir, hints

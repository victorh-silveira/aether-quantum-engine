"""Modificador adaptativo de pisos por estouro de volatilidade M15/M1."""

from __future__ import annotations


_VOL_BURST_MACRO_RATIO = 1.25
_VOL_BURST_MICRO_BB_WIDTH = 0.02
_VOL_BOOST_MANDATORY_SCORE = 0.65
_VOL_BOOST_MIN_EDGE = 0.03


def _macro_vol_ratio(metrics: dict) -> float:
    """Le vol_ratio macro M15 com fallback para indicadores micro."""
    macro = metrics.get("macro_indicators")
    if isinstance(macro, dict) and macro.get("vol_ratio") is not None:
        return float(macro["vol_ratio"])
    indicators = metrics.get("indicators") or {}
    return float(indicators.get("vol_ratio", 1.0))


def _micro_bb_width(metrics: dict) -> float:
    """Le bb_width micro M1 dos indicadores de execucao."""
    indicators = metrics.get("indicators") or {}
    return float(indicators.get("bb_width", 0.0))


def volatility_burst_active(metrics: dict) -> bool:
    """True quando M15 macro e M1 micro indicam explosao de volatilidade direcional."""
    return _macro_vol_ratio(metrics) > _VOL_BURST_MACRO_RATIO and _micro_bb_width(metrics) > _VOL_BURST_MICRO_BB_WIDTH


def apply_volatility_vol_booster(
    metrics: dict,
    *,
    mandatory_min_trade_score: float,
    min_edge_execute: float,
) -> tuple[float, float]:
    """Afrouxa pisos de score e edge em regime de estouro de volatilidade."""
    if not volatility_burst_active(metrics):
        return float(mandatory_min_trade_score), float(min_edge_execute)
    metrics["volatility_vol_booster"] = True
    return (
        min(float(mandatory_min_trade_score), _VOL_BOOST_MANDATORY_SCORE),
        min(float(min_edge_execute), _VOL_BOOST_MIN_EDGE),
    )

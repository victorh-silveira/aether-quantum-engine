"""Modificador adaptativo de pisos por estouro de volatilidade M15/M1."""

from __future__ import annotations

from src.application.services.deep_learning.dl_indicator_config import load_indicator_config_from_settings
from src.application.services.execution_runtime_config import resolve_volatility_booster_config


def _macro_vol_ratio(metrics: dict) -> float:
    """Le vol_ratio macro M15 com fallback para indicadores micro."""
    macro = metrics.get("macro_indicators")
    if isinstance(macro, dict) and macro.get("vol_ratio") is not None:
        return float(macro["vol_ratio"])
    indicators = metrics.get("indicators") or {}
    return float(indicators.get("vol_ratio", 1.0))


def _micro_bb_width(metrics: dict) -> float:
    """Le bb_width micro M5 dos indicadores de execucao."""
    indicators = metrics.get("indicators") or {}
    return float(indicators.get("bb_width", 0.0))


def volatility_burst_active(metrics: dict, *, vol_burst: dict | None = None) -> bool:
    """True quando M15 macro e M1 micro indicam explosao de volatilidade direcional."""
    cfg = vol_burst if isinstance(vol_burst, dict) else load_indicator_config_from_settings()["vol_burst"]
    return _macro_vol_ratio(metrics) > float(cfg["macro_vol_ratio_min"]) and _micro_bb_width(metrics) > float(
        cfg["micro_bb_width_min"]
    )


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
    boost = resolve_volatility_booster_config()
    return (
        min(float(mandatory_min_trade_score), float(boost["mandatory_score"])),
        min(float(min_edge_execute), float(boost["min_edge"])),
    )

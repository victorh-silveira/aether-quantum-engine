"""Ajuste dinamico de peso DL via correlacao cruzada entre indices."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_runtime_config import resolve_cross_corr_config


def _squeeze_active(metrics: dict) -> bool:
    """Resolve ou aplica  squeeze active."""
    indicators = metrics.get("indicators") or {}
    vol_ratio = float(indicators["vol_ratio"]) if "vol_ratio" in indicators else 1.0
    if vol_ratio < float(resolve_cross_corr_config()["squeeze_vol_ratio_max"]):
        return True
    return bool(metrics.get("bb_squeeze") or metrics.get("squeeze_active"))


def _weak_consensus(metrics: dict, *, min_margin: float) -> bool:
    """Indica consenso direcional fraco."""
    margin = float(metrics.get("direction_margin", 0.0))
    return margin + 1e-9 < float(min_margin)


def adjust_dl_weight_with_correlation(
    weights: dict[str, float],
    symbol: str,
    metrics: dict,
    corr_matrix: dict[tuple[str, str], float],
    *,
    anchor: str = "1HZ75V",
    min_margin: float | None = None,
) -> dict[str, float]:
    """Resolve ou aplica adjust dl weight with correlation."""
    cfg = resolve_cross_corr_config()
    margin_floor = float(cfg["min_margin"]) if min_margin is None else float(min_margin)
    if not _squeeze_active(metrics) and not _weak_consensus(metrics, min_margin=margin_floor):
        return weights
    merged = dict(weights)
    base = float(merged["dl_raw_weight"]) if "dl_raw_weight" in merged else float(cfg["dl_raw_weight"])
    corr = float(corr_matrix.get((str(symbol), str(anchor)), corr_matrix.get((str(anchor), str(symbol)), 0.0)))
    corr = max(-1.0, min(1.0, corr))
    if abs(corr) > float(cfg["high_corr_abs"]):
        retention = max(
            float(cfg["high_corr_retention_floor"]), 1.0 - abs(corr) * float(cfg["high_corr_retention_coef"])
        )
        merged["dl_raw_weight"] = base * retention
        metrics["cross_corr_dl_retention"] = retention
        metrics["cross_corr_anchor"] = anchor
    elif abs(corr) < float(cfg["low_corr_abs"]):
        merged["dl_raw_weight"] = min(float(cfg["low_corr_weight_cap"]), base * float(cfg["low_corr_weight_boost"]))
    return merged


def cached_correlation_matrix(orch: Any) -> dict[tuple[str, str], float]:
    """Retorna matriz de correlacao do cache local do orquestrador."""
    cached = getattr(orch, "_corr_matrix_cache", None)
    if isinstance(cached, dict):
        return cached
    return {}

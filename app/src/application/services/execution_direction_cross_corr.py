"""Ajuste dinamico de peso DL via correlacao cruzada entre indices."""

from __future__ import annotations

from typing import Any


def _squeeze_active(metrics: dict) -> bool:
    """Indica regime de compressao de volatilidade."""
    indicators = metrics.get("indicators") or {}
    vol_ratio = float(indicators.get("vol_ratio", 1.0))
    if vol_ratio < 0.85:
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
    anchor: str = "R_10",
    min_margin: float = 0.05,
) -> dict[str, float]:
    """Mistura dl_raw_weight com correlacao cruzada em squeeze ou consenso fraco."""
    if not _squeeze_active(metrics) and not _weak_consensus(metrics, min_margin=min_margin):
        return weights
    merged = dict(weights)
    base = float(merged.get("dl_raw_weight", 0.45))
    corr = float(corr_matrix.get((str(symbol), str(anchor)), corr_matrix.get((str(anchor), str(symbol)), 0.0)))
    corr = max(-1.0, min(1.0, corr))
    if abs(corr) > 0.55:
        retention = max(0.35, 1.0 - abs(corr) * 0.45)
        merged["dl_raw_weight"] = base * retention
        metrics["cross_corr_dl_retention"] = retention
        metrics["cross_corr_anchor"] = anchor
    elif abs(corr) < 0.25:
        merged["dl_raw_weight"] = min(0.65, base * 1.08)
    return merged


def cached_correlation_matrix(orch: Any) -> dict[tuple[str, str], float]:
    """Retorna matriz de correlacao do cache local do orquestrador."""
    cached = getattr(orch, "_corr_matrix_cache", None)
    if isinstance(cached, dict):
        return cached
    return {}

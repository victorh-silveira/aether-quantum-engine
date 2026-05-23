"""Selecao de um unico cluster US ou EU por ciclo conforme macro risk_on/risk_off."""

from __future__ import annotations

from typing import Any


def exclusive_cluster_by_macro_enabled(orch: Any) -> bool:
    """Indica se apenas um cluster regional deve executar por ciclo."""
    strategy = orch.config.get("strategy", {}) if isinstance(getattr(orch, "config", None), dict) else {}
    corr = strategy.get("correlation") if isinstance(strategy.get("correlation"), dict) else {}
    macro = strategy.get("macro") if isinstance(strategy.get("macro"), dict) else {}
    if "exclusive_cluster_by_macro" in corr:
        return bool(corr.get("exclusive_cluster_by_macro"))
    if "exclusive_cluster_by_macro" in macro:
        return bool(macro.get("exclusive_cluster_by_macro"))
    return True


def _macro_tag(metrics: dict[str, Any]) -> str:
    """Normaliza tag macro da decisao ancora para comparacao."""
    raw = metrics.get("macro_sentiment") or metrics.get("macro_confluence_tag") or ""
    return str(raw).strip().lower()


_MACRO_TAG_REGION: dict[str, str] = {
    "risk_on": "us",
    "risk_off": "eu",
    "divergence_us_leads": "us",
    "divergence_eu_leads": "eu",
}


def _macro_strength(metrics: dict[str, Any], region: str) -> float:
    """Le forca quantitativa US ou EU anexada as metricas da ancora."""
    raw = metrics.get("macro_us_strength_quant") if region == "us" else metrics.get("macro_eu_strength_quant")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def resolve_exclusive_cluster_region(metrics: dict[str, Any]) -> str | None:
    """Retorna us, eu ou None quando nao ha cluster exclusivo (empate indefinido)."""
    tag = _macro_tag(metrics)
    mapped = _MACRO_TAG_REGION.get(tag)
    if mapped:
        return mapped
    if tag in ("", "indefinido"):
        us_s = _macro_strength(metrics, "us")
        eu_s = _macro_strength(metrics, "eu")
        if us_s > eu_s:
            return "us"
        if eu_s > us_s:
            return "eu"
    return None


def cluster_region_for_symbol(
    sym: str,
    *,
    us_targets: tuple[str, ...] | list[str],
    eu_targets: tuple[str, ...] | list[str],
) -> str | None:
    """Classifica simbolo como us, eu ou None se nao pertence a cluster."""
    if sym in us_targets:
        return "us"
    if sym in eu_targets:
        return "eu"
    return None

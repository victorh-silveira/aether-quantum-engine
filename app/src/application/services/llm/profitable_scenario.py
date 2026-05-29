"""Filtro de cenarios lucrativos calibrados no backtest (tag macro + indice)."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.macro_config import resolve_macro_config


def _normalize_symbol_map(raw: Any) -> dict[str, tuple[str, ...]] | None:
    """Normaliza mapa tag macro -> simbolos permitidos."""
    if not isinstance(raw, dict):
        return None
    out: dict[str, tuple[str, ...]] = {}
    for key, value in raw.items():
        tag = str(key).strip()
        if not tag or not isinstance(value, (list, tuple)):
            continue
        syms = tuple(str(x).strip() for x in value if str(x).strip())
        if syms:
            out[tag] = syms
    return out or None


def _normalize_conviction_map(raw: Any) -> dict[str, float] | None:
    """Normaliza mapa tag macro -> piso minimo de conviccao."""
    if not isinstance(raw, dict):
        return None
    out: dict[str, float] = {}
    for key, value in raw.items():
        tag = str(key).strip()
        if not tag:
            continue
        try:
            out[tag] = max(0.0, min(0.99, float(value)))
        except (TypeError, ValueError):
            continue
    return out or None


def resolve_profitable_scenario_config(macro_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Mescla macro base com allowlist de simbolos e conviccao por tag."""
    base = resolve_macro_config(macro_cfg if isinstance(macro_cfg, dict) else None)
    raw = macro_cfg if isinstance(macro_cfg, dict) else {}
    sym_map = _normalize_symbol_map(raw.get("allowed_cluster_symbols_by_tag"))
    conv_map = _normalize_conviction_map(raw.get("min_conviction_by_tag"))
    merged = dict(base)
    if sym_map is not None:
        merged["allowed_cluster_symbols_by_tag"] = sym_map
    if conv_map is not None:
        merged["min_conviction_by_tag"] = conv_map
    return merged


def cluster_symbol_allowed_for_tag(
    macro_cfg: dict[str, Any] | None,
    *,
    macro_tag: str,
    symbol: str,
) -> bool:
    """True quando o simbolo esta na allowlist da tag macro ou nao ha restricao."""
    cfg = resolve_profitable_scenario_config(macro_cfg)
    allow = cfg.get("allowed_cluster_symbols_by_tag")
    if not isinstance(allow, dict):
        return True
    syms = allow.get(macro_tag)
    if not syms:
        return True
    return str(symbol) in syms


def min_conviction_for_macro_tag(
    macro_cfg: dict[str, Any] | None,
    *,
    macro_tag: str,
    base_floor: float,
) -> float:
    """Retorna piso de conviccao da tag macro ou o piso global do motor."""
    cfg = resolve_profitable_scenario_config(macro_cfg)
    by_tag = cfg.get("min_conviction_by_tag")
    if isinstance(by_tag, dict) and macro_tag in by_tag:
        return max(base_floor, float(by_tag[macro_tag]))
    return base_floor

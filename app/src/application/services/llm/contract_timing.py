"""Duracao de contrato e espacamento pos-liquidacao para refresh de cluster."""

from __future__ import annotations

from typing import Any


def contract_duration_seconds(config: dict[str, Any] | None) -> float:
    """Retorna duracao do contrato Rise/Fall em segundos a partir de risk_management.params."""
    root = config if isinstance(config, dict) else {}
    rm = root.get("risk_management") if isinstance(root.get("risk_management"), dict) else {}
    params = rm.get("params") if isinstance(rm.get("params"), dict) else {}
    try:
        dur = float(params.get("duration", 1))
    except (TypeError, ValueError):
        dur = 1.0
    unit = str(params.get("duration_unit", "m")).strip().lower()
    if unit in ("s", "sec", "second", "seconds"):
        return max(1.0, dur)
    if unit in ("h", "hour", "hours"):
        return max(60.0, dur * 3600.0)
    return max(60.0, dur * 60.0)


def resolve_refresh_entry_spacing_seconds(
    orch_cfg: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> float:
    """Espacamento pos-liquidacao para refresh/exec alinhado ao contrato M1."""
    cfg = orch_cfg if isinstance(orch_cfg, dict) else {}
    contract_sec = contract_duration_seconds(config)
    if bool(cfg.get("entry_spacing_follows_contract", True)):
        breath = float(cfg.get("post_settlement_breath_seconds", max(5.0, contract_sec * 0.1)))
        margin = float(cfg.get("cluster_refresh_entry_spacing_seconds", max(3.0, contract_sec * 0.05)))
        cap = float(cfg.get("cluster_refresh_spacing_cap_seconds", contract_sec + 12.0))
        return min(cap, max(0.0, breath + margin))
    breath = float(cfg.get("post_settlement_breath_seconds", 60))
    margin = float(cfg.get("cluster_refresh_entry_spacing_seconds", 30))
    return max(0.0, breath + margin)


def resolve_post_settlement_breath_seconds(
    orch_cfg: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> float:
    """Folego asyncio pos-liquidacao antes do proximo ciclo."""
    cfg = orch_cfg if isinstance(orch_cfg, dict) else {}
    contract_sec = contract_duration_seconds(config)
    if bool(cfg.get("breath_follows_contract", True)):
        explicit = cfg.get("post_settlement_breath_seconds")
        default = max(5.0, min(12.0, contract_sec * 0.12))
        if explicit is None:
            return default
        return max(0.0, min(float(explicit), contract_sec * 0.2))
    return max(0.0, float(cfg.get("post_settlement_breath_seconds", 60)))

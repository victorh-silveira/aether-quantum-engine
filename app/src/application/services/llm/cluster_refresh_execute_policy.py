"""Politica hibrida de execucao no cluster_refresh (quant em divergencia, LLM em risk)."""

from __future__ import annotations

import time
from typing import Any

from src.application.services.llm.macro_cluster_align import quant_trade_direction
from src.domain.models.trade import TradeDirection


def _normalize_quant_tags(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, (list, tuple)):
        tags = tuple(str(x).strip() for x in raw if str(x).strip())
        return tags or ("divergence_us_leads", "divergence_eu_leads")
    return ("divergence_us_leads", "divergence_eu_leads")


def resolve_cluster_refresh_policy(orch_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = orch_cfg if isinstance(orch_cfg, dict) else {}
    breath = float(cfg.get("post_settlement_breath_seconds", 60))
    margin = float(cfg.get("cluster_refresh_entry_spacing_seconds", 30))
    return {
        "global_refresh_execute": bool(cfg.get("cluster_refresh_execute_enabled", False)),
        "quant_refresh_execute": bool(cfg.get("cluster_refresh_execute_on_quant_validate", True)),
        "quant_tags": _normalize_quant_tags(cfg.get("cluster_refresh_quant_tags")),
        "entry_spacing_seconds": max(0.0, breath + margin),
    }


def _m5_aligns_with_direction(metrics: dict[str, Any], symbol: str, direction: TradeDirection) -> bool:
    raw_map = metrics.get("index_m5_dir_by_symbol")
    if not isinstance(raw_map, dict):
        return False
    micro = str(raw_map.get(symbol) or "")
    implied = quant_trade_direction(micro)
    return implied is not None and implied == direction


def _metrics_quant_validated(metrics: dict[str, Any], direction: TradeDirection) -> bool:
    if bool(metrics.get("llm_statarb_dir_corrected")):
        return True
    if str(metrics.get("decision_source") or "") == "cluster_statarb_dir":
        return True
    sym = str(metrics.get("cluster_target_sym") or "")
    return bool(sym and _m5_aligns_with_direction(metrics, sym, direction))


def entry_is_quant_validated(entry: dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return False
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict) or not metrics.get("execute"):
        return False
    direction = entry.get("direction")
    if not isinstance(direction, TradeDirection):
        return False
    return _metrics_quant_validated(metrics, direction)


def cluster_entry_spacing_allows(orch: Any, *, now_epoch: float | None = None) -> tuple[bool, str]:
    state = getattr(orch, "state", None)
    active = getattr(state, "active_contracts", None) if state is not None else None
    if active:
        return False, "active_contract_open"
    cfg = resolve_cluster_refresh_policy(
        orch.config.get("orchestrator") if isinstance(getattr(orch, "config", None), dict) else None
    )
    spacing = float(cfg["entry_spacing_seconds"])
    if spacing <= 0:
        return True, ""
    last_end = float(getattr(orch, "_last_cluster_cycle_end", 0.0) or 0.0)
    if last_end <= 0:
        return True, ""
    now = float(now_epoch if now_epoch is not None else time.time())
    if (now - last_end) < spacing:
        return False, "post_settlement_spacing"
    return True, ""


def macro_tag_from_decisions(decisions: dict[str, dict], anchor_sym: str) -> str:
    entry = decisions.get(anchor_sym) if isinstance(decisions, dict) else None
    if isinstance(entry, dict):
        metrics = entry.get("metrics")
        if isinstance(metrics, dict):
            return str(metrics.get("macro_sentiment") or metrics.get("macro_confluence_tag") or "")
    for entry in (decisions or {}).values():
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics")
        if isinstance(metrics, dict):
            tag = str(metrics.get("macro_sentiment") or metrics.get("macro_confluence_tag") or "")
            if tag:
                return tag
    return ""


def any_quant_validated_cluster_entry(
    decisions: dict[str, dict],
    *,
    anchor_sym: str,
) -> bool:
    if not isinstance(decisions, dict):
        return False
    for sym, entry in decisions.items():
        if sym == anchor_sym:
            continue
        if entry_is_quant_validated(entry):
            return True
    return False


def _divergence_refresh_gate(
    orch: Any,
    decisions: dict[str, dict],
    policy: dict[str, Any],
    *,
    now_epoch: float | None,
) -> tuple[bool, str]:
    anchor = str(getattr(orch, "anchor", "frxEURUSD") or "frxEURUSD")
    macro_tag = macro_tag_from_decisions(decisions, anchor)
    if macro_tag not in policy["quant_tags"]:
        return False, "risk_regime_requires_fresh_llm"
    if not policy["quant_refresh_execute"]:
        return False, "cluster_refresh_execute_disabled"
    if not any_quant_validated_cluster_entry(decisions, anchor_sym=anchor):
        return False, "divergence_refresh_no_quant_edge"
    spacing_ok, spacing_reason = cluster_entry_spacing_allows(orch, now_epoch=now_epoch)
    if not spacing_ok:
        return False, spacing_reason
    return True, "quant_refresh_ok"


def cluster_refresh_may_execute(
    orch: Any,
    decisions: dict[str, dict],
    *,
    refresh_without_llm: bool,
    now_epoch: float | None = None,
) -> tuple[bool, str]:
    if not refresh_without_llm:
        return True, ""
    orch_cfg = orch.config.get("orchestrator") if isinstance(getattr(orch, "config", None), dict) else {}
    policy = resolve_cluster_refresh_policy(orch_cfg)
    if policy["global_refresh_execute"]:
        return True, "global_refresh_execute"
    return _divergence_refresh_gate(orch, decisions, policy, now_epoch=now_epoch)

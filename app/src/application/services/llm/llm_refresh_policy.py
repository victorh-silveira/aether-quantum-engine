"""Politica de refresh da decisao LLM (alinhada ao backtest gemini tag_change)."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.global_macro_confluence import MacroSnapshot
from src.application.services.llm.llm_cluster_guards import cluster_conviction_floor
from src.application.services.llm.macro_config import resolve_macro_config
from src.domain.models.trade import TradeDirection


SCHEDULE_ALWAYS = "always"
SCHEDULE_TAG_CHANGE = "tag_change"
SCHEDULE_DAILY = "daily"


def resolve_llm_refresh_schedule(config: dict[str, Any]) -> str:
    """Resolve agenda de consulta LLM a partir de llm.refresh_schedule."""
    llm = config.get("llm", {}) if isinstance(config.get("llm"), dict) else {}
    raw = str(llm.get("refresh_schedule", SCHEDULE_TAG_CHANGE)).strip().lower()
    if raw in (SCHEDULE_ALWAYS, SCHEDULE_TAG_CHANGE, SCHEDULE_DAILY):
        return raw
    return SCHEDULE_TAG_CHANGE


def resolve_llm_refresh_interval_seconds(config: dict[str, Any]) -> float:
    """Intervalo maximo sem nova consulta Gemini (0 = desligado)."""
    llm = config.get("llm", {}) if isinstance(config.get("llm"), dict) else {}
    hours = float(llm.get("refresh_interval_hours", 0))
    return max(0.0, hours * 3600.0)


def anchor_cached_decision_valid(cached: dict[str, Any] | None, anchor_sym: str) -> bool:
    """True quando cache da ancora tem TradeDirection valida."""
    if not isinstance(cached, dict):
        return False
    entry = cached.get(anchor_sym)
    if not isinstance(entry, dict):
        return False
    return isinstance(entry.get("direction"), TradeDirection)


def clear_cluster_execute_on_cached_decisions(cached: dict[str, dict], anchor_sym: str) -> dict[str, dict]:
    """Zera execute nos indices ao reutilizar cache sem LLM fresca."""
    out: dict[str, dict] = {}
    for sym, entry in cached.items():
        if not isinstance(entry, dict):
            continue
        if sym == anchor_sym:
            out[sym] = dict(entry)
            continue
        e = dict(entry)
        m = dict(e.get("metrics") or {})
        m["execute"] = False
        if str(m.get("llm_block_reason") or "") == "allowed":
            m["llm_block_reason"] = "cache_no_fresh_llm"
        e["metrics"] = m
        out[sym] = e
    return out


def macro_tag_allows_llm_call(snapshot: MacroSnapshot, macro_cfg: dict[str, Any] | None) -> tuple[bool, str]:
    """False quando a tag macro bloqueia consulta; indefinido sempre consulta Gemini."""
    tag = str(snapshot.tag or "").strip()
    if tag in ("", "indefinido"):
        return True, ""
    cfg = resolve_macro_config(macro_cfg)
    allowed = cfg.get("allowed_execute_tags")
    if allowed and tag not in allowed:
        return False, f"MACRO_SKIP tag={snapshot.tag}"
    return True, ""


def cached_anchor_conviction_below_execute_floor(
    cached: dict[str, Any] | None,
    anchor_sym: str,
    macro_tag: str,
    macro_cfg: dict[str, Any] | None,
    base_floor: float,
) -> bool:
    """True quando cache da ancora tem conviccao abaixo do piso de entrada do cluster."""
    if not isinstance(cached, dict):
        return False
    entry = cached.get(anchor_sym)
    if not isinstance(entry, dict):
        return False
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    try:
        conviction = float(metrics.get("conviction", 0))
    except (TypeError, ValueError):
        return False
    if conviction <= 0:
        return False
    floor = cluster_conviction_floor(
        macro_cfg,
        macro_tag=macro_tag,
        base_floor=base_floor,
        llm_cluster_explicit=True,
    )
    return conviction < floor


def should_refresh_llm_decision(
    *,
    schedule: str,
    current_tag: str,
    last_tag: str | None,
    has_cached_decisions: bool,
    last_refresh_epoch: float | None = None,
    now_epoch: float | None = None,
    refresh_interval_seconds: float = 0.0,
) -> bool:
    """True quando a API Gemini deve ser consultada neste ciclo."""
    sched = (schedule or SCHEDULE_TAG_CHANGE).strip().lower()
    if sched == SCHEDULE_ALWAYS:
        return True
    if sched == SCHEDULE_DAILY:
        return True
    if not has_cached_decisions:
        return True
    interval = max(0.0, float(refresh_interval_seconds))
    if (
        interval > 0
        and last_refresh_epoch is not None
        and now_epoch is not None
        and float(now_epoch) - float(last_refresh_epoch) >= interval
    ):
        return True
    if str(current_tag).strip() in ("", "indefinido"):
        return True
    return str(current_tag) != str(last_tag or "")

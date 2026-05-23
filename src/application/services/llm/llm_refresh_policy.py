"""Politica de refresh da decisao LLM (alinhada ao backtest gemini tag_change)."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.global_macro_confluence import MacroSnapshot
from src.application.services.llm.macro_config import resolve_macro_config


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


def macro_tag_allows_llm_call(snapshot: MacroSnapshot, macro_cfg: dict[str, Any] | None) -> tuple[bool, str]:
    """False quando a tag macro atual nao esta em allowed_execute_tags."""
    cfg = resolve_macro_config(macro_cfg)
    allowed = cfg.get("allowed_execute_tags")
    if allowed and snapshot.tag not in allowed:
        return False, f"MACRO_SKIP tag={snapshot.tag}"
    return True, ""


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
    return str(current_tag) != str(last_tag or "")

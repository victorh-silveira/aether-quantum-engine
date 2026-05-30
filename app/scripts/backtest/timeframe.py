"""Granularidade primaria do backtest a partir de data_handler.granularity."""

from __future__ import annotations

from typing import Any


def primary_granularity_seconds(config: dict[str, Any]) -> int:
    """Segundos da vela primaria (data_handler.granularity ou 900)."""
    dh = config.get("data_handler", {})
    if isinstance(dh, dict) and dh.get("granularity"):
        return max(60, int(dh["granularity"]))
    return 900


def bars_per_day(granularity_seconds: int) -> int:
    """Quantidade de velas primarias em um dia UTC."""
    return max(1, 86400 // max(60, int(granularity_seconds)))


def bar_minutes(granularity_seconds: int) -> int:
    """Duracao em minutos de cada vela primaria."""
    return max(1, int(granularity_seconds) // 60)


def micro_granularity_seconds(config: dict[str, Any]) -> int:
    """Segundos da serie micro (fallback macro ou trigger LLM)."""
    macro = config.get("strategy", {}).get("macro", {})
    if isinstance(macro, dict) and macro.get("cluster_fallback_granularity_seconds"):
        return max(60, int(macro["cluster_fallback_granularity_seconds"]))
    llm = config.get("llm", {})
    if isinstance(llm, dict) and llm.get("micro_trigger_granularity_seconds"):
        return max(60, int(llm["micro_trigger_granularity_seconds"]))
    return 60

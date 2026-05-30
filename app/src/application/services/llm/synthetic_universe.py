"""Universo padrao de indices sinteticos Deriv (volatilidade M1/M5)."""

from __future__ import annotations

from src.application.services.llm.contract_timing import (
    contract_duration_seconds,
    resolve_post_settlement_breath_seconds,
    resolve_refresh_entry_spacing_seconds,
)


DEFAULT_ANCHOR = "R_100"

DEFAULT_US_CLUSTER = ("R_10", "R_25", "R_50")

DEFAULT_EU_CLUSTER = ("R_75", "1HZ50V", "1HZ100V")

DEFAULT_SYMBOLS = (
    DEFAULT_ANCHOR,
    *DEFAULT_US_CLUSTER,
    *DEFAULT_EU_CLUSTER,
)

M5_GRANULARITY_SECONDS = 300

M1_GRANULARITY_SECONDS = 60

DEFAULT_CLUSTER_LABELS = {
    "us": ("VOL10", "VOL25", "VOL50"),
    "eu": ("VOL75", "VOL50_1S", "VOL100_1S"),
}


def default_strategy_clusters() -> dict[str, list[str]]:
    """Retorna clusters US/EU padrao de volatilidade sintetica."""
    return {
        "us": list(DEFAULT_US_CLUSTER),
        "eu": list(DEFAULT_EU_CLUSTER),
    }


def resolve_anchor(config: dict) -> str:
    """Resolve simbolo ancora a partir de strategy.correlation ou config.anchor."""
    strategy = config.get("strategy", {})
    correlation = strategy.get("correlation", {}) if isinstance(strategy, dict) else {}
    if isinstance(correlation, dict) and correlation.get("anchor"):
        return str(correlation["anchor"])
    return str(config.get("anchor", DEFAULT_ANCHOR))


__all__ = [
    "DEFAULT_ANCHOR",
    "DEFAULT_EU_CLUSTER",
    "DEFAULT_US_CLUSTER",
    "DEFAULT_SYMBOLS",
    "M1_GRANULARITY_SECONDS",
    "M5_GRANULARITY_SECONDS",
    "DEFAULT_CLUSTER_LABELS",
    "contract_duration_seconds",
    "default_strategy_clusters",
    "resolve_anchor",
    "resolve_post_settlement_breath_seconds",
    "resolve_refresh_entry_spacing_seconds",
]

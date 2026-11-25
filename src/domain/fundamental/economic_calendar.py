"""Gate fundamentalista de calendario economico e controle de correlacao regional."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


EUROPE_BLOCK = frozenset({"OTC_GDAXI", "OTC_FCHI", "OTC_FTSE", "OTC_SX5E", "OTC_AEX", "OTC_SSMI"})
US_BLOCK = frozenset({"OTC_SPC", "OTC_NDX", "OTC_DJI"})
ASIA_PAC_BLOCK = frozenset({"OTC_N225", "OTC_AS51", "OTC_HSI"})

_CURRENCY_TO_BLOCK: dict[str, str] = {
    "EUR": "EUROPE",
    "GBP": "EUROPE",
    "CHF": "EUROPE",
    "JPY": "ASIA_PAC",
    "AUD": "ASIA_PAC",
    "NZD": "ASIA_PAC",
    "CNY": "ASIA_PAC",
    "CAD": "US",
}


@dataclass(frozen=True)
class EconomicEvent:
    """Evento macroeconomico agendado com timestamp UTC e classificacao de impacto."""

    title: str
    country: str
    epoch: int
    impact: str
    forecast: str = ""
    previous: str = ""


@dataclass(frozen=True)
class CalendarPolicy:
    """Politica deterministica de blackout pre-release e overlap de contrato."""

    enabled: bool = True
    impacts: frozenset[str] = frozenset({"HIGH"})
    usd_medium_enabled: bool = False
    usd_global: bool = True
    pre_minutes: int = 15
    post_minutes: int = 0
    fail_open: bool = True


def symbol_regional_block(symbol: str) -> str:
    """Retorna o bloco regional canonico do ativo OTC ou o proprio simbolo."""
    if symbol in EUROPE_BLOCK:
        return "EUROPE"
    if symbol in US_BLOCK:
        return "US"
    if symbol in ASIA_PAC_BLOCK:
        return "ASIA_PAC"
    return symbol


def policy_from_config(config: dict[str, Any] | None) -> CalendarPolicy:
    """Constroi CalendarPolicy imutavel a partir do dicionario SSOT de configuracao."""
    cfg = (config or {}).get("economic_calendar") or {}
    raw_impacts = cfg.get("impacts") or ("High",)
    impacts = frozenset(str(item).strip().upper() for item in raw_impacts if str(item).strip())
    return CalendarPolicy(
        enabled=bool(cfg.get("enabled", False)),
        impacts=impacts or frozenset({"HIGH"}),
        usd_medium_enabled=bool(cfg.get("usd_medium_enabled", False)),
        usd_global=bool(cfg.get("usd_global", True)),
        pre_minutes=max(0, int(cfg.get("pre_minutes", 15))),
        post_minutes=max(0, int(cfg.get("post_minutes", 0))),
        fail_open=bool(cfg.get("fail_open", True)),
    )


def event_affects_symbol(event: EconomicEvent, symbol: str, policy: CalendarPolicy) -> bool:
    """Verifica se o evento macro atende impacto e escopo regional do ativo."""
    country = event.country.strip().upper()
    impact = event.impact.strip().upper()
    impact_ok = impact in policy.impacts or (policy.usd_medium_enabled and country == "USD" and impact == "MEDIUM")
    if not impact_ok:
        return False
    if country == "USD":
        return True if policy.usd_global else symbol_regional_block(symbol) == "US"
    target_block = _CURRENCY_TO_BLOCK.get(country)
    return bool(target_block and symbol_regional_block(symbol) == target_block)


def evaluate_calendar_veto(
    symbol: str,
    entry_epoch: int,
    period_seconds: int,
    events: Iterable[EconomicEvent],
    policy: CalendarPolicy,
) -> tuple[bool, str]:
    """Veta entrada se o contrato cruzar ou anteceder evento macro qualificado."""
    if not policy.enabled or period_seconds <= 0:
        return False, ""
    window_start = entry_epoch - policy.post_minutes * 60
    window_end = entry_epoch + period_seconds + policy.pre_minutes * 60
    for event in events:
        if not event_affects_symbol(event, symbol, policy):
            continue
        if window_start <= event.epoch <= window_end:
            return True, f"veto_calendario_macro:{event.country.upper()}:{event.impact.upper()}"
    return False, ""


def apply_regional_block_cap(ranked_orders: list[tuple], max_per_block: int) -> list[tuple]:
    """Limita ordens simultaneas no mesmo bloco regional preservando a ordem de ranking."""
    if max_per_block <= 0:
        return list(ranked_orders)
    counts: dict[str, int] = {}
    selected: list[tuple] = []
    for order in ranked_orders:
        block = symbol_regional_block(str(order[0]))
        used = counts.get(block, 0)
        if used >= max_per_block:
            continue
        counts[block] = used + 1
        selected.append(order)
    return selected

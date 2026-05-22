"""Guardrails Medallion: StatArb/HMM ajustam conviccao sem vetar a LLM."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.global_macro_confluence import MacroSnapshot, resolve_macro_config
from src.domain.models.trade import TradeDirection


def divergence_leader_strength(snapshot: MacroSnapshot, tag: str) -> float:
    """Retorna a forca do cluster lider em tags de divergencia transatlantica."""
    if tag == "divergence_us_leads":
        return float(snapshot.us_strength)
    if tag == "divergence_eu_leads":
        return float(snapshot.eu_strength)
    return 0.0


def _statarb_z_for_symbol(snapshot: MacroSnapshot, sym: str) -> float | None:
    """Retorna o Z-Score StatArb do simbolo ou None se indisponivel."""
    spreads = getattr(snapshot, "statarb_spreads", None)
    if not spreads or sym not in spreads:
        return None
    return float(spreads[sym])


def _apply_statarb_intelligence(
    direction: TradeDirection | None,
    conviction: float,
    z: float,
    z_threshold: float,
    hmm_state: int,
) -> tuple[TradeDirection | None, float, bool, list[str]]:
    """Ajusta conviccao via StatArb/HMM sem vetar a direcao da LLM."""
    note_parts: list[str] = []
    guard_applied = False
    if hmm_state == 0:
        if z < -z_threshold and direction == TradeDirection.CALL:
            return direction, min(0.99, conviction + 0.10), True, [f"STATARB_INTEL boost CALL (Z={z:.2f})"]
        if z > z_threshold and direction == TradeDirection.PUT:
            return direction, min(0.99, conviction + 0.10), True, [f"STATARB_INTEL boost PUT (Z={z:.2f})"]
        if abs(z) > z_threshold:
            note_parts.append(f"STATARB_INTEL spread_diverge (Z={z:.2f})")
            return direction, max(0.0, conviction - 0.04), True, note_parts
    elif abs(z) > z_threshold:
        note_parts.append(f"STATARB_INTEL trending_caution (Z={z:.2f})")
        return direction, max(0.0, conviction - 0.03), True, note_parts
    return direction, conviction, guard_applied, note_parts


def apply_macro_confluence_guard(
    direction: TradeDirection | None,
    conviction: float,
    snapshot: MacroSnapshot | None,
    macro_cfg: dict[str, Any] | None,
    sym: str | None = None,
) -> tuple[TradeDirection | None, float, bool, str, bool]:
    """Preserva decisao LLM e anota ajustes quantitativos StatArb/HMM."""
    if direction is None or snapshot is None:
        return direction, conviction, False, "", True

    cfg = resolve_macro_config(macro_cfg)
    note_parts: list[str] = []
    guard_applied = False
    conviction_final = conviction
    if sym:
        z = _statarb_z_for_symbol(snapshot, sym)
        if z is not None:
            z_threshold = float(cfg.get("statarb_z_threshold", 2.5))
            hmm_state = int(getattr(snapshot, "hmm_state", 0))
            direction, conviction_final, sa_applied, sa_notes = _apply_statarb_intelligence(
                direction, conviction_final, z, z_threshold, hmm_state
            )
            if sa_applied:
                guard_applied = True
            note_parts.extend(sa_notes)
    if snapshot.tag.startswith("divergence"):
        guard_applied = True
        note_parts.append(f"MACRO_INTEL tag={snapshot.tag} llm_sovereign=1")
    return direction, conviction_final, guard_applied, " | ".join(note_parts), True

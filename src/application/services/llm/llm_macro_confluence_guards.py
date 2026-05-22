"""Guardrails Medallion: StatArb/HMM e macro filtram entradas de baixa qualidade."""

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


def _statarb_misaligned(direction: TradeDirection, z: float, z_threshold: float, hmm_state: int) -> bool:
    """True quando Z contradiz a direcao em regime de reversao (HMM 0)."""
    if hmm_state != 0:
        return False
    if direction == TradeDirection.CALL and z > z_threshold:
        return True
    return direction == TradeDirection.PUT and z < -z_threshold


def _apply_statarb_intelligence(
    direction: TradeDirection | None,
    conviction: float,
    z: float,
    z_threshold: float,
    hmm_state: int,
) -> tuple[TradeDirection | None, float, bool, list[str], bool]:
    """Ajusta conviccao via StatArb/HMM; pode vetar execucao."""
    note_parts: list[str] = []
    guard_applied = False
    if _statarb_misaligned(direction, z, z_threshold, hmm_state):
        note_parts.append(f"STATARB_VETO misalign (Z={z:.2f})")
        return direction, conviction, True, note_parts, False
    if hmm_state == 0:
        if z < -z_threshold and direction == TradeDirection.CALL:
            return direction, min(0.99, conviction + 0.08), True, [f"STATARB_INTEL boost CALL (Z={z:.2f})"], True
        if z > z_threshold and direction == TradeDirection.PUT:
            return direction, min(0.99, conviction + 0.08), True, [f"STATARB_INTEL boost PUT (Z={z:.2f})"], True
        if abs(z) > z_threshold:
            note_parts.append(f"STATARB_INTEL spread_diverge (Z={z:.2f})")
            return direction, max(0.0, conviction - 0.06), True, note_parts, False
    elif abs(z) > z_threshold:
        note_parts.append(f"STATARB_INTEL trending_caution (Z={z:.2f})")
        return direction, max(0.0, conviction - 0.05), True, note_parts, True
    return direction, conviction, guard_applied, note_parts, True


def _cluster_floor_ok(snapshot: MacroSnapshot, floor: float) -> bool:
    """True se ambos os clusters atingem o piso de forca."""
    return min(float(snapshot.us_strength), float(snapshot.eu_strength)) >= floor


def _divergence_macro_ok(snapshot: MacroSnapshot, cfg: dict[str, Any], conviction: float) -> tuple[bool, list[str]]:
    """Valida tag de divergencia transatlantica."""
    notes: list[str] = []
    tag = snapshot.tag
    floor = float(cfg["confluence_conviction_floor"])
    div_min = float(cfg.get("divergence_min_leader_strength", floor + 0.05))
    div_cap = float(cfg["divergence_max_conviction"])
    gap_min = float(cfg.get("divergence_min_strength_gap", 0.05))
    leader_s = divergence_leader_strength(snapshot, tag)
    us_s = float(snapshot.us_strength)
    eu_s = float(snapshot.eu_strength)
    if leader_s < div_min:
        notes.append(f"MACRO_VETO divergence_leader<{div_min:.2f}")
        return False, notes
    if tag == "divergence_us_leads" and (us_s - eu_s) < gap_min:
        notes.append("MACRO_VETO divergence_us_gap")
        return False, notes
    if tag == "divergence_eu_leads" and (eu_s - us_s) < gap_min:
        notes.append("MACRO_VETO divergence_eu_gap")
        return False, notes
    if conviction > div_cap:
        notes.append(f"MACRO_CAP divergence_conviction>{div_cap:.2f}")
    return True, notes


def _indefinido_macro_ok(snapshot: MacroSnapshot, cfg: dict[str, Any]) -> tuple[bool, list[str]]:
    """Valida tag indefinido com lider regional claro."""
    notes: list[str] = []
    floor = float(cfg["confluence_conviction_floor"])
    indef_min = float(cfg.get("indefinido_min_leader_strength", floor + 0.03))
    gap_min = float(cfg.get("indefinido_min_strength_gap", 0.06))
    us_s = float(snapshot.us_strength)
    eu_s = float(snapshot.eu_strength)
    leader = max(us_s, eu_s)
    gap = abs(us_s - eu_s)
    if leader < indef_min or gap < gap_min:
        notes.append("MACRO_VETO indefinido_no_leader")
        return False, notes
    return True, notes


def _macro_tag_allows_execute(
    snapshot: MacroSnapshot, cfg: dict[str, Any], conviction: float
) -> tuple[bool, list[str]]:
    """Valida tag macro e forca regional antes de executar."""
    notes: list[str] = []
    tag = snapshot.tag
    floor = float(cfg["confluence_conviction_floor"])
    hmm_min = float(cfg.get("assert_min_hmm_prob", 0.0))
    allowed = False

    if float(getattr(snapshot, "hmm_prob", 1.0)) < hmm_min:
        notes.append(f"MACRO_VETO hmm_prob<{hmm_min:.2f}")
    elif tag == "risk_on":
        allowed = _cluster_floor_ok(snapshot, floor)
        if not allowed:
            notes.append("MACRO_VETO risk_on_weak_clusters")
    elif tag == "risk_off":
        allowed = _cluster_floor_ok(snapshot, floor)
        if not allowed:
            notes.append("MACRO_VETO risk_off_weak_clusters")
    elif tag.startswith("divergence"):
        allowed, notes = _divergence_macro_ok(snapshot, cfg, conviction)
    elif tag == "indefinido":
        allowed, notes = _indefinido_macro_ok(snapshot, cfg)
    else:
        notes.append(f"MACRO_VETO unknown_tag={tag}")

    return allowed, notes


def apply_macro_confluence_guard(
    direction: TradeDirection | None,
    conviction: float,
    snapshot: MacroSnapshot | None,
    macro_cfg: dict[str, Any] | None,
    sym: str | None = None,
) -> tuple[TradeDirection | None, float, bool, str, bool]:
    """Filtra entradas fracas; preserva direcao LLM quando executavel."""
    if direction is None or snapshot is None:
        return direction, conviction, False, "", False

    cfg = resolve_macro_config(macro_cfg)
    note_parts: list[str] = []
    guard_applied = False
    conviction_final = conviction

    macro_ok, macro_notes = _macro_tag_allows_execute(snapshot, cfg, conviction_final)
    note_parts.extend(macro_notes)
    if not macro_ok:
        return direction, conviction_final, True, " | ".join(note_parts), False

    if snapshot.tag.startswith("divergence"):
        div_cap = float(cfg["divergence_max_conviction"])
        conviction_final = min(conviction_final, div_cap)
        guard_applied = True

    if sym:
        z = _statarb_z_for_symbol(snapshot, sym)
        if z is not None:
            z_threshold = float(cfg.get("statarb_z_threshold", 2.5))
            hmm_state = int(getattr(snapshot, "hmm_state", 0))
            direction, conviction_final, sa_applied, sa_notes, sa_ok = _apply_statarb_intelligence(
                direction, conviction_final, z, z_threshold, hmm_state
            )
            if sa_applied:
                guard_applied = True
            note_parts.extend(sa_notes)
            if not sa_ok:
                return direction, conviction_final, guard_applied, " | ".join(note_parts), False

    if snapshot.tag.startswith("divergence"):
        guard_applied = True
        note_parts.append(f"MACRO_INTEL tag={snapshot.tag}")

    return direction, conviction_final, guard_applied, " | ".join(note_parts), True

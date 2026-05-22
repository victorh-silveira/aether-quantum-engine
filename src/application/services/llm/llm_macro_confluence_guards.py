"""Guardrails de confluencia macro transatlantica e StatArb Medallion."""

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


def _calculate_mcs_and_conviction(
    direction: TradeDirection,
    conviction: float,
    snapshot: MacroSnapshot,
    bias: str,
    tag: str,
    avg_strength: float,
) -> float:
    """Calcula o Macro Confluence Score (MCS) e a conviccao final."""
    if snapshot.cluster_status == "":
        return conviction

    if tag in ("risk_on", "risk_off"):
        mcs = 0.80 + 0.15 * avg_strength
        if bias in ("CALL", "PUT"):
            quant_dir = TradeDirection.CALL if bias == "CALL" else TradeDirection.PUT
            if direction == quant_dir:
                mcs += 0.04
        mcs = min(0.99, max(0.0, mcs))
    elif tag.startswith("divergence"):
        leader_s = divergence_leader_strength(snapshot, tag)
        mcs = 0.70 + 0.20 * leader_s
        if bias in ("CALL", "PUT"):
            quant_dir = TradeDirection.CALL if bias == "CALL" else TradeDirection.PUT
            if direction == quant_dir:
                mcs += 0.04
        mcs = min(0.99, max(0.0, mcs))
    else:
        mcs = 0.52 + 0.08 * avg_strength

    return (conviction * 0.3) + (mcs * 0.7)


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


def _apply_statarb_legacy(
    direction: TradeDirection | None,
    conviction: float,
    z: float,
    z_threshold: float,
    hmm_state: int,
) -> tuple[TradeDirection | None, float, bool, list[str], bool]:
    """Aplica StatArb legado com boost ou bloqueio de execucao."""
    note_parts: list[str] = []
    guard_applied = False
    execute_ok = True
    if hmm_state == 0:
        if z < -z_threshold:
            if direction == TradeDirection.CALL:
                return direction, min(0.99, conviction + 0.15), False, [f"STATARB_BOOST CALL (Z={z:.2f})"], True
            if direction == TradeDirection.PUT:
                note_parts.append(f"STATARB_BLOCK conflict PUT vs StatArb CALL (Z={z:.2f})")
                return None, conviction, True, note_parts, False
        if z > z_threshold:
            if direction == TradeDirection.PUT:
                return direction, min(0.99, conviction + 0.15), False, [f"STATARB_BOOST PUT (Z={z:.2f})"], True
            if direction == TradeDirection.CALL:
                note_parts.append(f"STATARB_BLOCK conflict CALL vs StatArb PUT (Z={z:.2f})")
                return None, conviction, True, note_parts, False
    elif abs(z) > z_threshold:
        note_parts.append(f"STATARB_TREND_BLOCK volatility_regime=TRENDING (Z={z:.2f})")
        return None, conviction, True, note_parts, False
    return direction, conviction, guard_applied, note_parts, execute_ok


def _apply_statarb_guard(
    direction: TradeDirection | None,
    conviction: float,
    snapshot: MacroSnapshot,
    cfg: dict[str, Any],
    sym: str,
    *,
    intelligence_only: bool = False,
) -> tuple[TradeDirection | None, float, bool, list[str], bool]:
    """Aplica StatArb e HMM: modo inteligencia ajusta conviccao; modo legado pode vetar."""
    z = _statarb_z_for_symbol(snapshot, sym)
    if z is None:
        return direction, conviction, False, [], True
    z_threshold = float(cfg.get("statarb_z_threshold", 2.5))
    hmm_state = int(getattr(snapshot, "hmm_state", 0))
    if intelligence_only:
        out_dir, out_conv, applied, notes = _apply_statarb_intelligence(
            direction, conviction, z, z_threshold, hmm_state
        )
        return out_dir, out_conv, applied, notes, True
    out_dir, out_conv, applied, notes, execute_ok = _apply_statarb_legacy(
        direction, conviction, z, z_threshold, hmm_state
    )
    return out_dir, out_conv, applied, notes, execute_ok


def _apply_macro_intelligence_guard(
    direction: TradeDirection | None,
    conviction: float,
    snapshot: MacroSnapshot,
    cfg: dict[str, Any],
    sym: str | None,
) -> tuple[TradeDirection | None, float, bool, str, bool]:
    """Modo inteligencia pura: preserva decisao LLM e anota sinais quant."""
    note_parts: list[str] = []
    guard_applied = False
    conviction_final = conviction
    if sym:
        direction, conviction_final, sa_applied, sa_notes, _ = _apply_statarb_guard(
            direction,
            conviction_final,
            snapshot,
            cfg,
            sym,
            intelligence_only=True,
        )
        if sa_applied:
            guard_applied = True
        note_parts.extend(sa_notes)
    if snapshot.tag.startswith("divergence"):
        guard_applied = True
        note_parts.append(f"MACRO_INTEL tag={snapshot.tag} llm_sovereign=1")
    return direction, conviction_final, guard_applied, " | ".join(note_parts), True


def _apply_macro_legacy_vetos(
    direction: TradeDirection | None,
    conviction: float,
    snapshot: MacroSnapshot,
    cfg: dict[str, Any],
    bias: str,
    tag: str,
) -> tuple[TradeDirection | None, float, bool, list[str], bool]:
    """Aplica tetos e vetos macro do modo legado."""
    note_parts: list[str] = []
    guard_applied = False
    execute_ok = True
    if tag.startswith("divergence") and cfg["divergence_blocks_execution"]:
        guard_applied = True
        cap = float(cfg["divergence_max_conviction"])
        conviction = min(conviction, cap)
        note_parts.append(f"MACRO_DIV cap={cap:.2f}")
    if not cfg["align_eurusd_with_confluence"] or bias not in ("CALL", "PUT"):
        return direction, conviction, guard_applied, note_parts, execute_ok
    quant_dir = TradeDirection.CALL if bias == "CALL" else TradeDirection.PUT
    floor = float(cfg["confluence_conviction_floor"])
    min_strength = min(float(snapshot.us_strength), float(snapshot.eu_strength))
    strength_gate = divergence_leader_strength(snapshot, tag) if tag.startswith("divergence") else min_strength
    if (
        strength_gate >= floor
        and direction is not None
        and direction != quant_dir
        and (tag in ("risk_on", "risk_off") or tag.startswith("divergence"))
    ):
        guard_applied = True
        execute_ok = False
        direction = None
        veto_tag = "MACRO_DIV_VETO" if tag.startswith("divergence") else "MACRO_ALIGN"
        note_parts.append(f"{veto_tag} block bias={bias} tag={tag}")
    return direction, conviction, guard_applied, note_parts, execute_ok


def apply_macro_confluence_guard(
    direction: TradeDirection | None,
    conviction: float,
    snapshot: MacroSnapshot | None,
    macro_cfg: dict[str, Any] | None,
    sym: str | None = None,
) -> tuple[TradeDirection | None, float, bool, str, bool]:
    """Aplica confluencia macro: inteligencia pura preserva decisao LLM com sinais quant."""
    if direction is None or snapshot is None:
        return direction, conviction, False, "", True

    cfg = resolve_macro_config(macro_cfg)
    intelligence_only = bool(cfg.get("macro_intelligence_only", False))

    if intelligence_only:
        return _apply_macro_intelligence_guard(direction, conviction, snapshot, cfg, sym)

    tag = snapshot.tag
    bias = snapshot.eurusd_bias
    avg_strength = (float(snapshot.us_strength) + float(snapshot.eu_strength)) / 2.0
    conviction_final = _calculate_mcs_and_conviction(direction, conviction, snapshot, bias, tag, avg_strength)
    note_parts: list[str] = []
    guard_applied = False
    execute_ok = True
    if sym:
        direction, conviction_final, sa_applied, sa_notes, sa_ok = _apply_statarb_guard(
            direction, conviction_final, snapshot, cfg, sym
        )
        if sa_applied:
            guard_applied = True
        if not sa_ok:
            execute_ok = False
        note_parts.extend(sa_notes)
    direction, conviction_final, veto_applied, veto_notes, veto_ok = _apply_macro_legacy_vetos(
        direction, conviction_final, snapshot, cfg, bias, tag
    )
    if veto_applied:
        guard_applied = True
    if not veto_ok:
        execute_ok = False
    note_parts.extend(veto_notes)
    return direction, conviction_final, guard_applied, " | ".join(note_parts), execute_ok

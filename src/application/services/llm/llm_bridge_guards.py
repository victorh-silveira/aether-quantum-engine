"""Guardrails e calculos MTF para o bridge LLM."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

import src.application.services.llm.indicators as ti
from src.application.services.llm.global_macro_confluence import MacroSnapshot, resolve_macro_config
from src.application.services.llm.llm_bridge_utils import score_token
from src.domain.models.trade import TradeDirection


TREND_FOLLOWING_ACTIVE = "TREND_FOLLOWING_ACTIVE"


_TRIPLE_X = re.compile(r"\b(alta|baixa|alto|baixo)\s*x\s*(alta|baixa|alto|baixo)\s*x\s*(alta|baixa|alto|baixo)")
_QUAD_X = re.compile(
    r"\b(alta|baixa|alto|baixo)\s*x\s*(alta|baixa|alto|baixo)\s*x\s*(alta|baixa|alto|baixo)\s*x\s*(alta|baixa|alto|baixo)"
)


def _mtf_alignment_structured(low: str) -> bool:
    """True se a linha parece alinhamento MTF estruturado."""
    return (
        ("m15:" in low and "m5:" in low and "m3:" in low)
        or ("m30:" in low and "m5:" in low and "m1:" in low)
        or (("h1:" in low or "m30:" in low or "d1:" in low) and "m15:" in low and "m5:" in low and "m1:" in low)
        or bool(_TRIPLE_X.search(low))
        or bool(_QUAD_X.search(low))
    )


def mtf_score_from_alignment(mtf_alignment: str) -> int:
    """Calcula score MTF direto da string de alinhamento para telemetria."""
    lowered = (mtf_alignment or "").lower()
    tokens = re.findall(r"alta|alto|baixa|baixo|bullish|bearish|bull|bear", lowered)
    if len(tokens) >= 4:
        a, b, c, d = tokens[0], tokens[1], tokens[2], tokens[3]
        return score_token(a) + score_token(b) + (2 * score_token(c)) + (3 * score_token(d))
    if len(tokens) >= 3:
        macro_raw, m5_raw, micro_raw = tokens[0], tokens[1], tokens[2]
        return score_token(macro_raw) + (2 * score_token(m5_raw)) + (3 * score_token(micro_raw))
    return 0


def merge_mtf_scores(score_desc: int, score_align: int, mtf_alignment: str = "") -> int:
    """Combina scores apenas para telemetria."""
    low = (mtf_alignment or "").lower()
    structured = _mtf_alignment_structured(low)
    if score_align == 0:
        return 0 if structured else score_desc
    return score_align


def mtf_score(macro_desc: str, structure_desc: str, swing_desc: str, trigger_desc: str) -> int:
    """Calcula score ponderado de tendencia para fins de telemetria."""
    return (
        score_token(macro_desc)
        + score_token(structure_desc)
        + (2 * score_token(swing_desc))
        + (3 * score_token(trigger_desc))
    )


def rsi_exhaustion_execution_gate(
    direction: TradeDirection,
    rsi_m1_last: float | None,
    *,
    rsi_block_call_above: float,
    rsi_block_put_below: float,
    gate_enabled: bool,
) -> tuple[TradeDirection, str | None]:
    """Aplica limiar RSI na execucao: modo operacao continua (ignora travas de WAIT)."""
    _ = (rsi_m1_last, rsi_block_call_above, rsi_block_put_below, gate_enabled)
    return direction, TREND_FOLLOWING_ACTIVE


def is_sawtooth_pattern(mtf_d: str) -> bool:
    """Detecta padrao P/M/P/M ou M/P/M/P no token compacto MTF para telemetria."""
    raw = (mtf_d or "").strip().upper()
    if "/" in raw:
        parts = [p for p in raw.split("/") if p in ("P", "M")]
        if len(parts) >= 4:
            p1, p2, p3, p4 = parts[:4]
            return p1 != p2 != p3 != p4
    return False


def is_highly_divergent_mtf(macro_d: str, struct_d: str, swing_d: str, trigger_d: str) -> bool:
    """True se houver conflito frontal entre as camadas macro e micro para telemetria."""
    ma = score_token(macro_d)
    st = score_token(struct_d)
    sw = score_token(swing_d)
    tr = score_token(trigger_d)
    total_score = ma + st + sw + tr
    return abs(total_score) <= 1 and ((ma + st > 1 and sw + tr < -1) or (ma + st < -1 and sw + tr > 1))


def invert_call_put(direction: TradeDirection) -> TradeDirection:
    """Troca CALL por PUT e vice-versa."""
    if direction == TradeDirection.CALL:
        return TradeDirection.PUT
    if direction == TradeDirection.PUT:
        return TradeDirection.CALL
    return direction


def is_overextended(closes: list[float], window: int = 20) -> bool:
    """True se o preco atual for uma anomalia estatistica (Z-Score > 3.0)."""
    c = np.asarray(closes, dtype=np.float64)
    z = ti._z_score_last(c, window)
    if z is None:
        return False
    return abs(z) > 3.0


def _calculate_mcs_and_conviction(
    direction: TradeDirection,
    conviction: float,
    snapshot: MacroSnapshot,
    bias: str,
    tag: str,
    avg_strength: float,
) -> float:
    """Calcula o Macro Confluence Score (MCS) e a convicção final."""
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
        mcs = 0.58 + 0.07 * avg_strength
    else:
        mcs = 0.52 + 0.08 * avg_strength

    return (conviction * 0.3) + (mcs * 0.7)


def _apply_statarb_guard(
    direction: TradeDirection | None,
    conviction: float,
    snapshot: MacroSnapshot,
    cfg: dict[str, Any],
    sym: str,
) -> tuple[TradeDirection | None, float, bool, list[str], bool]:
    """Aplica o filtro de espalhamento StatArb e HMM de volatilidade para o ativo."""
    note_parts: list[str] = []
    guard_applied = False
    execute_ok = True

    if not getattr(snapshot, "statarb_spreads", None) or sym not in snapshot.statarb_spreads:
        return direction, conviction, guard_applied, note_parts, execute_ok

    z = float(snapshot.statarb_spreads[sym])
    z_threshold = float(cfg.get("statarb_z_threshold", 2.5))
    hmm_state = int(getattr(snapshot, "hmm_state", 0))

    if hmm_state == 0:  # MEAN_REVERSION
        if z < -z_threshold:  # Undervalued: CALL pressure
            if direction == TradeDirection.CALL:
                conviction = min(0.99, conviction + 0.15)
                note_parts.append(f"STATARB_BOOST CALL (Z={z:.2f})")
            elif direction == TradeDirection.PUT:
                guard_applied = True
                execute_ok = False
                direction = None
                note_parts.append(f"STATARB_BLOCK conflict PUT vs StatArb CALL (Z={z:.2f})")
        elif z > z_threshold:  # Overvalued: PUT pressure
            if direction == TradeDirection.PUT:
                conviction = min(0.99, conviction + 0.15)
                note_parts.append(f"STATARB_BOOST PUT (Z={z:.2f})")
            elif direction == TradeDirection.CALL:
                guard_applied = True
                execute_ok = False
                direction = None
                note_parts.append(f"STATARB_BLOCK conflict CALL vs StatArb PUT (Z={z:.2f})")
    elif abs(z) > z_threshold:  # TRENDING
        guard_applied = True
        execute_ok = False
        direction = None
        note_parts.append(f"STATARB_TREND_BLOCK volatility_regime=TRENDING (Z={z:.2f})")

    return direction, conviction, guard_applied, note_parts, execute_ok


def apply_macro_confluence_guard(
    direction: TradeDirection | None,
    conviction: float,
    snapshot: MacroSnapshot | None,
    macro_cfg: dict[str, Any] | None,
    sym: str | None = None,
) -> tuple[TradeDirection | None, float, bool, str, bool]:
    """Aplica guardrails e calculo de confluencia macro transatlantica dinamica."""
    if direction is None or snapshot is None:
        return direction, conviction, False, "", True

    cfg = resolve_macro_config(macro_cfg)
    tag = snapshot.tag
    bias = snapshot.eurusd_bias

    us_strength = float(snapshot.us_strength)
    eu_strength = float(snapshot.eu_strength)
    avg_strength = (us_strength + eu_strength) / 2.0

    # Calculate conviction
    conviction_final = _calculate_mcs_and_conviction(direction, conviction, snapshot, bias, tag, avg_strength)

    # Apply guardrails
    note_parts: list[str] = []
    guard_applied = False
    execute_ok = True

    # 1. StatArb Cointegration Spread and Pacemaking HMM guards
    if sym:
        direction, conviction_final, sa_applied, sa_notes, sa_ok = _apply_statarb_guard(
            direction, conviction_final, snapshot, cfg, sym
        )
        if sa_applied:
            guard_applied = True
        if not sa_ok:
            execute_ok = False
        note_parts.extend(sa_notes)

    # 2. Existing standard confluence guardrails
    if tag.startswith("divergence") and cfg["divergence_blocks_execution"]:
        guard_applied = True
        cap = float(cfg["divergence_max_conviction"])
        conviction_final = min(conviction_final, cap)
        note_parts.append(f"MACRO_DIV cap={cap:.2f}")

    if cfg["align_eurusd_with_confluence"] and bias in ("CALL", "PUT"):
        quant_dir = TradeDirection.CALL if bias == "CALL" else TradeDirection.PUT
        floor = float(cfg["confluence_conviction_floor"])
        min_strength = min(us_strength, eu_strength)
        # Only block if we haven't already blocked by StatArb
        if (
            min_strength >= floor
            and direction is not None
            and direction != quant_dir
            and tag in ("risk_on", "risk_off")
        ):
            guard_applied = True
            execute_ok = False
            direction = None
            note_parts.append(f"MACRO_ALIGN block bias={bias}")

    note = " | ".join(note_parts)
    return direction, conviction_final, guard_applied, note, execute_ok

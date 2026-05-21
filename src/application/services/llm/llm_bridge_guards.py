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


def apply_macro_confluence_guard(
    direction: TradeDirection | None,
    conviction: float,
    snapshot: MacroSnapshot | None,
    macro_cfg: dict[str, Any] | None,
) -> tuple[TradeDirection | None, float, bool, str, bool]:
    """Aplica guardrails de confluencia macro transatlantica na decisao EURUSD."""
    if direction is None or snapshot is None:
        return direction, conviction, False, "", True

    cfg = resolve_macro_config(macro_cfg)
    tag = snapshot.tag
    bias = snapshot.eurusd_bias
    min_strength = min(float(snapshot.us_strength), float(snapshot.eu_strength))
    note_parts: list[str] = []
    guard_applied = False
    execute_ok = True

    if tag.startswith("divergence") and cfg["divergence_blocks_execution"]:
        guard_applied = True
        cap = float(cfg["divergence_max_conviction"])
        conviction = min(conviction, cap)
        note_parts.append(f"MACRO_DIV cap={cap:.2f}")

    if cfg["align_eurusd_with_confluence"] and bias in ("CALL", "PUT"):
        quant_dir = TradeDirection.CALL if bias == "CALL" else TradeDirection.PUT
        floor = float(cfg["confluence_conviction_floor"])
        if min_strength >= floor and direction != quant_dir and tag in ("risk_on", "risk_off"):
            guard_applied = True
            execute_ok = False
            direction = None
            note_parts.append(f"MACRO_ALIGN block bias={bias}")

    note = " | ".join(note_parts)
    return direction, conviction, guard_applied, note, execute_ok

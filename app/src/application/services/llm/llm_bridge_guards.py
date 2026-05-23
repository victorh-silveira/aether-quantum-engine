"""Guardrails e calculos MTF para o bridge LLM."""

from __future__ import annotations

import re

from src.application.services.llm.llm_bridge_utils import score_token
from src.application.services.llm.llm_macro_confluence_guards import apply_macro_confluence_guard


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


__all__ = [
    "apply_macro_confluence_guard",
    "merge_mtf_scores",
    "mtf_score",
    "mtf_score_from_alignment",
]

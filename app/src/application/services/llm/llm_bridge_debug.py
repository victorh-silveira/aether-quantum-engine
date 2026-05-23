"""Utilitarios de debug e formatacao para a ponte LLM."""

from __future__ import annotations

import re
from typing import Any

from src.application.services.llm.llm_bridge_guards import (
    merge_mtf_scores,
    mtf_score,
    mtf_score_from_alignment,
)
from src.domain.models.trade import TradeDirection


def _compact_debug_mtf(text: str, max_chars: int) -> str:
    """Compacta texto de alinhamento MTF para exibicao em log."""
    raw = str(text or "").replace("**", "").replace("\n", " ")
    s = " ".join(raw.split()).strip()
    s = re.sub(r"\b(D1|H4|H1|M30|M15|M5|M3|M1):\s*(?=\||$)", r"\1: indefinido ", s)
    if not s:
        return "-"  # pragma: no cover
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."  # pragma: no cover


def emit_direction_debug(
    orch: Any,
    *,
    runtime: dict[str, Any],
    direction_base: TradeDirection,
    direction_final: TradeDirection,
    conviction: float,
    macro_desc: str,
    structure_desc: str,
    swing_desc: str,
    trigger_desc: str,
    mtf_alignment: str,
    decision_source: str,
    adjusted: bool,
    exec_inverted: bool = False,
) -> None:
    """Emite linha objetiva de debug com fatores da direcao final."""
    cid = f"C{int(orch._active_cycle_id):04d}"
    score_desc = mtf_score(macro_desc, structure_desc, swing_desc, trigger_desc)
    score_align = mtf_score_from_alignment(mtf_alignment)
    score = merge_mtf_scores(score_desc, score_align, mtf_alignment)
    cap = min(120, max(48, int(runtime.get("logic_line_max_chars", 140))))
    mtf_disp = _compact_debug_mtf(mtf_alignment, cap)
    logger = orch.logger
    logger.debug(
        "[%s] DEBUG DIRECAO: base=%s | final=%s | conv=%.1f%% | score_mtf=%s | ajustado=%s | exec_inv=%s | fonte=%s | mtf=%s | modelo=%s",
        cid,
        direction_base.name if direction_base else "NONE",
        direction_final.name if direction_final else "NONE",
        float(conviction) * 100.0,
        score,
        "sim" if adjusted else "nao",
        "sim" if exec_inverted else "nao",
        decision_source or "-",
        mtf_disp,
        runtime["model"],
    )

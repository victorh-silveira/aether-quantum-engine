"""Guardrails de repeticao para o bridge LLM."""

from __future__ import annotations

import re
from typing import Any

from src.application.services.llm.llm_bridge_utils import trend_token
from src.domain.models.trade import TradeDirection


def can_repeat_same_direction(
    direction: TradeDirection,
    note: str,
    m3_desc: str,
    rsi_status: str,
    runtime: dict[str, Any],
) -> bool:
    """Valida repeticao de direcao com confirmacao tecnica extra."""
    directional = direction in (TradeDirection.CALL, TradeDirection.PUT)
    if not directional:
        return False
    strict_enabled = bool(runtime.get("same_direction_strict_enabled", False))
    if not strict_enabled:
        return True

    lowered_note = str(note or "").lower()
    lowered_rsi = str(rsi_status or "").lower()
    rsi_match = re.search(r"rsi\s*=\s*([0-9]+(?:\.[0-9]+)?)", str(m3_desc or ""), flags=re.IGNORECASE)
    rsi_value = float(rsi_match.group(1)) if rsi_match else None

    neutral_floor = float(runtime.get("same_direction_rsi_min", 40.0))
    neutral_ceiling = float(runtime.get("same_direction_rsi_max", 60.0))

    rsi_is_neutral = rsi_value is not None and neutral_floor <= rsi_value <= neutral_ceiling
    if rsi_value is None:
        rsi_is_neutral = (
            "normal" in lowered_rsi and "sobrecompra" not in lowered_rsi and "sobrevenda" not in lowered_rsi
        )

    confirms = True
    if bool(runtime.get("same_direction_require_m3_confirmation", True)):
        m3_tk = trend_token(m3_desc)
        confirms = (direction == TradeDirection.CALL and m3_tk == "alta") or (
            direction == TradeDirection.PUT and m3_tk == "baixa"
        )

    wick_ok = True
    if bool(runtime.get("same_direction_require_wick_confirmation", True)):
        wick_ok = any(token in lowered_note for token in ("pavio", "rejeicao", "rejeição", "wick", "pinbar"))

    has_extreme_zone = rsi_value is not None and (rsi_value >= 75.0 or rsi_value <= 25.0)
    return rsi_is_neutral and confirms and wick_ok and not has_extreme_zone


def choose_direction_without_wait(
    direction: TradeDirection | None,
    m15_desc: str,
    m5_desc: str,
    m3_desc: str,
    last_direction: TradeDirection,
    mtf_alignment: str = "",
    prompt: str = "",
) -> TradeDirection | None:
    """Preserva a decisao da LLM sem qualquer fallback local."""
    _ = (m15_desc, m5_desc, m3_desc, last_direction, mtf_alignment, prompt)
    return direction


def apply_repeat_direction_guard(
    direction: TradeDirection,
    note: str,
    metrics: dict[str, Any],
    _m3_d: str,
    _rsi_status: str,
    runtime: dict[str, Any],
    last_dir: TradeDirection,
    streak: int,
) -> tuple[TradeDirection, str, dict[str, Any], int]:
    """Atualiza contador de streak sem alterar a decisao original da LLM."""
    if direction not in (TradeDirection.CALL, TradeDirection.PUT):
        return direction, note, metrics, 0
    streak = streak + 1 if direction == last_dir else 1
    max_streak = int(runtime.get("max_same_direction_streak", 0))
    if max_streak > 0 and streak > max_streak:
        streak = max_streak
    return direction, note, metrics, streak

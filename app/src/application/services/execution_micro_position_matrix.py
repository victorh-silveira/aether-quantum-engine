"""Matriz de micro-posicionamento dinamico M1: topo, fundo e meio de canal de 60s."""

from __future__ import annotations

import math
from typing import Any

from src.application.services.execution_universal_regime_types import (
    MICRO_BB_LOWER_TRIGGER,
    MICRO_BB_UPPER_TRIGGER,
    MICRO_EXHAUSTION_OVERRIDE_SCORE,
    MICRO_LOW_CONSENSUS_MARGIN,
    MICRO_LOW_CONSENSUS_MIN_MINORITY,
    MICRO_MIDDLE_UNCERTAINTY_SCORE,
    MicroMatrixDecision,
    MicroPositionZone,
    RegimeState,
)
from src.domain.models.trade import TradeDirection


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Converte indicador micro para float finito com fallback defensivo."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _micro_levels(metrics: dict) -> tuple[float, float]:
    """Extrai keltner e bb_pct_b do bloco micro M1 com centro neutro por padrao."""
    indicators = metrics.get("indicators") or {}
    keltner = _safe_float(indicators.get("keltner"), 0.5)
    bb_pct_b = _safe_float(indicators.get("bb_pct_b"), 0.5)
    return keltner, bb_pct_b


def classify_micro_zone(metrics: dict, *, keltner_topo: float, keltner_fundo: float) -> MicroPositionZone:
    """Classifica o ultimo tick em topo, fundo ou meio de canal micro M1."""
    keltner, bb_pct_b = _micro_levels(metrics)
    if keltner + 1e-9 >= keltner_topo or bb_pct_b + 1e-9 >= MICRO_BB_UPPER_TRIGGER:
        return MicroPositionZone.UPPER_BOUNDARY
    if keltner <= keltner_fundo + 1e-9 or bb_pct_b <= MICRO_BB_LOWER_TRIGGER + 1e-9:
        return MicroPositionZone.LOWER_BOUNDARY
    return MicroPositionZone.FAIR_VALUE_MIDDLE


def is_low_consensus(metrics: dict) -> bool:
    """True quando CALL e PUT tem votos mistos de baixo consenso (ex.: 4x2 ou 3x3)."""
    call_votes = int(_safe_float(metrics.get("call_votes"), 0.0))
    put_votes = int(_safe_float(metrics.get("put_votes"), 0.0))
    if min(call_votes, put_votes) < MICRO_LOW_CONSENSUS_MIN_MINORITY:
        return False
    return abs(call_votes - put_votes) <= MICRO_LOW_CONSENSUS_MARGIN


def _boundary_decision(zone: MicroPositionZone, *, is_trend: bool, boost: float) -> MicroMatrixDecision:
    """Resolve rompimento assimetrico ou exaustao defensiva em barreira micro."""
    if is_trend:
        return MicroMatrixDecision(zone=zone, breakout_boost=boost)
    return MicroMatrixDecision(
        zone=zone,
        exhaustion_invert=True,
        score_override=MICRO_EXHAUSTION_OVERRIDE_SCORE,
    )


_DECISIVE_MACRO_REGIMES = (RegimeState.CLIMAX_EXHAUSTION, RegimeState.ENTROPIC_NOISE)


def _enters_boundary(zone: MicroPositionZone, dl_dir: TradeDirection, exec_dir: TradeDirection) -> bool:
    """True quando DL e execucao apontam para dentro da barreira micro saturada."""
    if zone == MicroPositionZone.UPPER_BOUNDARY:
        return dl_dir == TradeDirection.CALL and exec_dir == TradeDirection.CALL
    if zone == MicroPositionZone.LOWER_BOUNDARY:
        return dl_dir == TradeDirection.PUT and exec_dir == TradeDirection.PUT
    return False


def build_micro_decision(
    zone: MicroPositionZone,
    macro_regime: RegimeState | None,
    dl_dir: TradeDirection,
    exec_dir: TradeDirection,
    metrics: dict,
    *,
    breakout_boost: float,
) -> MicroMatrixDecision:
    """Cruza regime macro M15 com zona micro M1 para acao tatica de posicionamento."""
    if macro_regime in _DECISIVE_MACRO_REGIMES:
        return MicroMatrixDecision(zone=zone)
    if _enters_boundary(zone, dl_dir, exec_dir):
        return _boundary_decision(
            zone,
            is_trend=macro_regime == RegimeState.TREND_EXPANSION,
            boost=breakout_boost,
        )
    if zone == MicroPositionZone.FAIR_VALUE_MIDDLE and macro_regime is None and is_low_consensus(metrics):
        return MicroMatrixDecision(
            zone=zone,
            middle_uncertainty=True,
            score_override=MICRO_MIDDLE_UNCERTAINTY_SCORE,
        )
    return MicroMatrixDecision(zone=zone)

"""Coesao micro M1 para inversao condicional em COMPRESSION_TRAP."""

from __future__ import annotations

from src.application.services.execution_universal_regime_types import RegimeEvaluation, RegimeState
from src.domain.models.trade import TradeDirection


COMPRESSION_TRAP_BB_WIDTH_MAX = 0.01


def micro_bb_width(metrics: dict) -> float:
    """Largura relativa das bandas de Bollinger no horizonte micro M1."""
    indicators = metrics.get("indicators") or {}
    raw = indicators.get("bb_width")
    if raw is None:
        return 1.0
    return float(raw)


def enforce_compression_trap_micro_bb_cohesion(
    exec_dir: TradeDirection,
    dl_dir: TradeDirection,
    metrics: dict,
    evaluation: RegimeEvaluation,
) -> TradeDirection:
    """So consome inversao de COMPRESSION_TRAP quando o micro M1 confirma compressao."""
    if evaluation.regime != RegimeState.COMPRESSION_TRAP:
        return exec_dir
    if not evaluation.direction_inverted:
        return exec_dir
    if micro_bb_width(metrics) + 1e-9 < COMPRESSION_TRAP_BB_WIDTH_MAX:
        return exec_dir
    metrics["compression_trap_bb_veto"] = True
    metrics["direction_inverted"] = False
    metrics["compression_trap_inverted"] = False
    metrics["resolved_direction"] = dl_dir.name
    metrics["exec_direction"] = dl_dir.name
    return dl_dir

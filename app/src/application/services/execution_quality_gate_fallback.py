"""Bloqueio de fallback obrigatorio quando o quality gate veta candidatos em recovery."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_quality_gate import read_risk_session_state


_QUALITY_TECHNICAL_BLOCKS = frozenset({"data", "predict_error", "training"})

__all__ = ["cluster_quality_gate_blocks_mandatory_fallback"]


def _entry_viable_for_quality_fallback(entry: dict) -> bool:
    """Indica se o entry tem sinal DL tecnico minimo para avaliar bloqueio de fallback."""
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        return False
    if metrics.get("deploy_ok") is False:
        return False
    if str(metrics.get("gate_reason") or "") in _QUALITY_TECHNICAL_BLOCKS:
        return False
    if entry.get("direction") is not None:
        return True
    return metrics.get("calibrated_prob") is not None or metrics.get("raw_prob") is not None


def cluster_quality_gate_blocks_mandatory_fallback(
    decisions: dict,
    *,
    exec_cfg: dict,
    risk_manager: Any | None,
    trade_symbols: list[str] | tuple[str, ...],
) -> bool:
    """Impede fallback obrigatorio quando recovery vetou todos os candidatos DL elegiveis."""
    session_linear, pending = read_risk_session_state(risk_manager)
    if session_linear <= 0 and pending <= 0.0:
        return False
    if not isinstance(decisions, dict):
        return False
    viable = 0
    rejected = 0
    for symbol in trade_symbols:
        entry = decisions.get(symbol)
        if not isinstance(entry, dict):
            continue
        if not _entry_viable_for_quality_fallback(entry):
            continue
        metrics = entry["metrics"]
        viable += 1
        if metrics.get("quality_guard_reject"):
            rejected += 1
    _ = exec_cfg
    return viable > 0 and rejected == viable

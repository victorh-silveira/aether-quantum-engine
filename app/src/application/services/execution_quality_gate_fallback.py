"""Bloqueio de fallback obrigatorio quando o quality gate veta candidatos em recovery."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_quality_gate import read_risk_session_state
from src.application.services.execution_quality_gate_microstructure import is_microstructure_starvation_reason
from src.domain.risk.stake_sizing import metric_float


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


def _hard_quality_reject_for_fallback(metrics: dict) -> bool:
    """True para veto de microestrutura ou meta Z < -0.20 em recovery."""
    if is_microstructure_starvation_reason(metrics.get("quality_gate_reason")):
        return True
    if not metrics.get("quality_guard_reject"):
        return False
    if metrics.get("meta_payoff_edge_zscore") is None and metrics.get("edge_zscore") is None:
        return False
    return metric_float(metrics, "meta_payoff_edge_zscore", "edge_zscore", default=0.0) < -0.20


def cluster_quality_gate_blocks_mandatory_fallback(
    decisions: dict,
    *,
    exec_cfg: dict,
    risk_manager: Any | None,
    trade_symbols: list[str] | tuple[str, ...],
) -> bool:
    """Impede fallback obrigatorio quando microestrutura ou recovery vetou todos os elegiveis."""
    if not isinstance(decisions, dict):
        return False
    viable = 0
    rejected = 0
    micro_rejected = 0
    for symbol in trade_symbols:
        entry = decisions.get(symbol)
        if not isinstance(entry, dict):
            continue
        if not _entry_viable_for_quality_fallback(entry):
            continue
        metrics = entry["metrics"]
        viable += 1
        if is_microstructure_starvation_reason(metrics.get("quality_gate_reason")):
            micro_rejected += 1
            rejected += 1
            continue
        if _hard_quality_reject_for_fallback(metrics):
            rejected += 1
    if viable > 0 and micro_rejected == viable:
        return True
    session_linear, pending = read_risk_session_state(risk_manager)
    if session_linear <= 0 and pending <= 0.0:
        return False
    _ = exec_cfg
    return viable > 0 and rejected == viable

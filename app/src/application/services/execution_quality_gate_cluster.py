"""Suspensao cooperativa do cluster quando o quality gate reprova candidatos."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_quality_gate import (
    format_quality_guard_log_message,
    passes_execution_quality,
    read_risk_session_state,
)
from src.application.services.meta_direction_flip import SIGNAL_SUSPENDED
from src.application.services.orchestrator.regime_freeze_yield import propagate_cluster_signal_suspended


__all__ = ["log_quality_guard_suspension", "quality_conviction_suspends_cluster"]


def log_quality_guard_suspension(orch: Any, *, reason: str = "") -> None:
    """Emite log deduplicado por ciclo quando o cluster e suspenso por qualidade."""
    cycle_id = int(getattr(orch, "_active_cycle_id", 0))
    logged_cycle = int(getattr(orch, "_quality_guard_logged_cycle_id", -1))
    if cycle_id <= logged_cycle:
        return
    orch._quality_guard_logged_cycle_id = cycle_id
    risk_manager = getattr(orch, "risk_manager", None)
    session_linear, pending_loss = read_risk_session_state(risk_manager)
    suspend_reason = reason or str(getattr(orch, "_quality_guard_last_reason", "") or "")
    if not suspend_reason:
        suspend_reason = "[Quality gate reject]"
    orch.logger.info(
        format_quality_guard_log_message(
            cycle_id,
            suspend_reason,
            linear=session_linear,
            pending_loss=pending_loss,
        ),
    )


def quality_conviction_suspends_cluster(orch: Any, decisions: dict) -> bool:
    """Retorna True quando algum simbolo falha no gate de alta conviccao."""
    if not isinstance(decisions, dict):
        return False
    exec_cfg = getattr(orch, "config", {}).get("orchestrator", {}).get("execution", {})
    risk_manager = getattr(orch, "risk_manager", None)
    suspended = False
    suspend_reason = ""
    for entry in decisions.values():
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            continue
        if passes_execution_quality(
            metrics,
            exec_cfg=exec_cfg if isinstance(exec_cfg, dict) else {},
            risk_manager=risk_manager,
        ):
            continue
        metrics["signal_status"] = SIGNAL_SUSPENDED
        metrics["quality_guard_reject"] = True
        suspended = True
        reason = metrics.get("quality_gate_reason")
        if isinstance(reason, str) and reason and not suspend_reason:
            suspend_reason = reason
    if not suspended:
        return False
    propagate_cluster_signal_suspended(decisions)
    orch._quality_guard_last_reason = suspend_reason
    log_quality_guard_suspension(orch, reason=suspend_reason)
    return True

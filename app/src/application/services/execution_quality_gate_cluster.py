"""Suspensao cooperativa do cluster quando o quality gate reprova candidatos."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.application.services.execution_quality_gate import (
    format_quality_guard_log_message,
    format_quality_guard_reject_message,
    passes_execution_quality,
    read_risk_session_state,
)
from src.application.services.execution_quality_gate_meta import evaluate_meta_payoff_quality
from src.application.services.log_dedupe import LogDeduper
from src.application.services.meta_direction_flip import SIGNAL_SUSPENDED
from src.application.services.orchestrator.regime_freeze_yield import propagate_cluster_signal_suspended


_MANDATORY_CONTINUOUS_REGIME = "mandatory_continuous"

__all__ = ["log_quality_guard_suspension", "quality_conviction_suspends_cluster"]


def _minute_bucket(orch: Any) -> str:
    """Resolve bucket de minuto para deduplicacao de logs do quality guard."""
    broker_time = getattr(orch, "_broker_server_time_utc", None)
    if broker_time is not None:
        return broker_time.strftime("%Y%m%d%H%M")
    return datetime.now(tz=UTC).strftime("%Y%m%d%H%M")


def _uses_meta_quality_gate(metrics: dict) -> bool:
    """Indica se o candidato deve ser avaliado pelo portao estatistico do meta-regressor."""
    return metrics.get("meta_payoff_edge_zscore") is not None or metrics.get("edge_zscore") is not None


def _mandatory_trade_each_cycle(exec_cfg: dict | None) -> bool:
    """Retorna se o motor opera em modo mandatario continuo."""
    if not isinstance(exec_cfg, dict):
        return True
    return bool(exec_cfg.get("mandatory_trade_each_cycle", True))


def _quality_guard_message(cycle_id: int, reason: str, *, linear: int, pending_loss: float) -> str:
    """Seleciona formato de log conforme o tipo de rejeicao do quality guard."""
    if "Meta Z-Score" in reason or reason in {"", "suspenso por meta-regressor"}:
        return format_quality_guard_reject_message(
            cycle_id,
            reason or "suspenso por meta-regressor",
            linear=linear,
            pending_loss=pending_loss,
        )
    return format_quality_guard_log_message(
        cycle_id,
        reason,
        linear=linear,
        pending_loss=pending_loss,
    )


def log_quality_guard_suspension(orch: Any, *, reason: str = "") -> None:
    """Emite log deduplicado por minuto quando o cluster e suspenso por qualidade."""
    cycle_id = int(getattr(orch, "_active_cycle_id", 0))
    risk_manager = getattr(orch, "risk_manager", None)
    session_linear, pending_loss = read_risk_session_state(risk_manager)
    suspend_reason = reason or str(getattr(orch, "_quality_guard_last_reason", "") or "")
    if not suspend_reason:
        suspend_reason = "suspenso por meta-regressor"
    message = _quality_guard_message(
        cycle_id,
        suspend_reason,
        linear=session_linear,
        pending_loss=pending_loss,
    )
    logger = getattr(orch, "logger", None)
    if logger is None:
        return
    LogDeduper(orch).log_quality_guard_cycle_minute(
        logger,
        cycle_id=cycle_id,
        minute_bucket=_minute_bucket(orch),
        message=message,
    )


def _annotate_quality_failure(metrics: dict, *, mandatory: bool) -> None:
    """Marca rejeicao de qualidade sem suspender sinal em modo mandatario."""
    metrics["quality_guard_reject"] = True
    if not mandatory:
        metrics["signal_status"] = SIGNAL_SUSPENDED


def quality_conviction_suspends_cluster(orch: Any, decisions: dict) -> bool:
    """Retorna True quando o cluster deve ser suspenso por falha no quality gate."""
    if not isinstance(decisions, dict):
        return False
    exec_cfg = getattr(orch, "config", {}).get("orchestrator", {}).get("execution", {})
    exec_cfg = exec_cfg if isinstance(exec_cfg, dict) else {}
    mandatory = _mandatory_trade_each_cycle(exec_cfg)
    risk_manager = getattr(orch, "risk_manager", None)
    skipped_cycles = int(getattr(orch, "_quality_skipped_cycles_counter", 0) or 0)
    any_fail = False
    any_pass = False
    meta_mode = False
    suspend_reason = ""
    for entry in decisions.values():
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            continue
        if metrics.get("deploy_ok") is False:
            continue
        if _uses_meta_quality_gate(metrics):
            meta_mode = True
            passed = evaluate_meta_payoff_quality(
                metrics,
                exec_cfg=exec_cfg,
                risk_manager=risk_manager,
                skipped_cycles_counter=skipped_cycles,
                orch=orch,
            )
        else:
            passed = passes_execution_quality(
                metrics,
                exec_cfg=exec_cfg,
                risk_manager=risk_manager,
                skipped_cycles_counter=skipped_cycles,
                orch=orch,
            )
        if passed:
            any_pass = True
            continue
        _annotate_quality_failure(metrics, mandatory=mandatory)
        any_fail = True
        reason = metrics.get("quality_gate_reason")
        if isinstance(reason, str) and reason and not suspend_reason:
            suspend_reason = reason
    if not any_fail:
        return False
    if meta_mode and any_pass:
        return False
    orch._quality_guard_last_reason = suspend_reason
    log_quality_guard_suspension(orch, reason=suspend_reason)
    if mandatory:
        orch._last_quality_gate_regime = _MANDATORY_CONTINUOUS_REGIME
        return False
    propagate_cluster_signal_suspended(decisions)
    return True

"""Telemetria do meta-regressor sem trava de execucao por payoff/Z-Score."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_quality_gate import (
    ensure_direction_margin,
    format_quality_guard_reject_message,
    quality_gate_params,
    read_risk_session_state,
    resolve_dynamic_quality_limits,
)
from src.application.services.log_dedupe import LogDeduper
from src.application.services.payoff_edge_zscore import (
    attach_payoff_edge_zscore_metrics,
)
from src.domain.risk.soft_recovery_policy import gbdt_waiver_skip_threshold_for_risk


def _meta_payoff_zscore(metrics: dict) -> float:
    """Le Z-Score meta-regressor anexado nas metricas do candidato."""
    raw = metrics.get("meta_payoff_edge_zscore")
    if raw is None:
        raw = metrics.get("edge_zscore")
    return float(raw or 0.0)


def resolve_min_meta_payoff_zscore(exec_cfg: dict | None) -> float:
    """Le limiar favoravel minimo de Z-Score meta-regressor."""
    params = quality_gate_params(exec_cfg or {})
    return float(params["min_meta_payoff_zscore"])


def ensure_meta_zscore_telemetry(
    metrics: dict,
    *,
    risk_manager: Any | None = None,
    linear: int | None = None,
    pending_loss_total: float | None = None,
) -> None:
    """Garante telemetria de Z-Score antes da avaliacao do portao."""
    if metrics.get("meta_payoff_edge_zscore") is not None or metrics.get("edge_zscore") is not None:
        return
    edge = float(metrics.get("predicted_payoff_edge", 0.0))
    _ = read_risk_session_state(
        risk_manager,
        linear=linear,
        pending_loss_total=pending_loss_total,
    )
    attach_payoff_edge_zscore_metrics(metrics, edge)


def meta_zscore_reject_reason(z_edge: float, *, min_z: float) -> str:
    """Formata motivo textual legado de rejeicao por Z-Score insuficiente."""
    return f"[Meta Z-Score {z_edge:.2f} < min {min_z:.2f}]"


def emit_quality_reject_log(orch: Any, *, cycle_id: int, reason: str, minute_bucket: str) -> None:
    """Emite log deduplicado legado de rejeicao por ciclo e bloco de minuto."""
    logger = getattr(orch, "logger", None)
    if logger is None:
        return
    risk_manager = getattr(orch, "risk_manager", None)
    session_linear, pending_loss = read_risk_session_state(risk_manager)
    message = format_quality_guard_reject_message(
        cycle_id,
        reason,
        linear=session_linear,
        pending_loss=pending_loss,
    )
    LogDeduper(orch).log_quality_guard_cycle_minute(
        logger,
        cycle_id=cycle_id,
        minute_bucket=minute_bucket,
        message=message,
    )


def evaluate_meta_payoff_quality(
    metrics: dict,
    *,
    exec_cfg: dict | None = None,
    risk_manager: Any | None = None,
    linear: int | None = None,
    pending_loss_total: float | None = None,
    min_direction_margin: float | None = None,
    min_payoff_edge: float | None = None,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
    log_reject: bool = False,
    minute_bucket: str | None = None,
) -> bool:
    """Telemetria meta apenas; trava de Meta Payoff/Z-Score desativada."""
    _ = (log_reject, minute_bucket, orch)
    limits = resolve_dynamic_quality_limits(
        exec_cfg or {},
        risk_manager=risk_manager,
        linear=linear,
        pending_loss_total=pending_loss_total,
        override_margin=min_direction_margin,
        override_edge=min_payoff_edge,
        skipped_cycles_counter=skipped_cycles_counter,
        orch=orch,
    )
    metrics["quality_gate_regime"] = str(limits.get("quality_regime", "regular"))
    metrics["quality_min_direction_margin"] = float(limits.get("min_direction_margin", 0.0))
    metrics["quality_min_payoff_edge"] = float(limits.get("min_payoff_edge", 0.0))
    metrics["quality_skipped_cycles_counter"] = float(limits.get("skipped_cycles_counter", 0.0))
    metrics["quality_starvation_decay_factor"] = float(limits.get("starvation_decay_factor", 1.0))
    metrics["recovery_relax_intensity"] = float(limits.get("recovery_relax_intensity", 0.0))
    ensure_direction_margin(metrics)
    ensure_meta_zscore_telemetry(
        metrics,
        risk_manager=risk_manager,
        linear=linear,
        pending_loss_total=pending_loss_total,
    )
    min_z = resolve_min_meta_payoff_zscore(exec_cfg)
    metrics["quality_min_meta_payoff_zscore"] = float(min_z)
    metrics["meta_payoff_edge_zscore_snapshot"] = _meta_payoff_zscore(metrics)
    metrics["gbdt_waiver_skip_threshold"] = int(gbdt_waiver_skip_threshold_for_risk(risk_manager))
    metrics["execution_gate_state"] = "meta_payoff_gate_disabled"
    metrics.pop("regime_skip_cycle", None)
    metrics.pop("quality_guard_reject", None)
    metrics.pop("quality_gate_reason", None)
    return True

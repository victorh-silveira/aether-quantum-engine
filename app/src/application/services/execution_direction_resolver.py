"""Motor de direcao TCN com telemetria meta-regressor (sem vetos de sinal)."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_direction_checks import (
    infer_dl_direction,
    initial_direction_checks,
    is_technically_blocked,
    seed_direction_metrics,
    sync_entry_metrics,
)
from src.application.services.execution_quality_gate_margin import ensure_direction_margin
from src.application.services.force_trade_mode import force_trade_every_cycle
from src.application.services.live_signal_metrics import apply_live_calib_drift_soft, attach_live_signal_metrics
from src.application.services.meta_classifier_stacking import resolve_meta_payoff_edge
from src.application.services.meta_payoff_regression import apply_meta_regression_edge
from src.application.services.payoff_edge_zscore import attach_payoff_edge_zscore_metrics
from src.domain.models.trade import TradeDirection


__all__ = (
    "infer_dl_direction",
    "is_technically_blocked",
    "resolve_execution_direction",
)


def _stamp_direction_resolved_cycle(entry: dict, cycle_id: int) -> None:
    """Marca o ciclo em que a direcao foi resolvida."""
    metrics = entry.setdefault("metrics", {})
    if isinstance(metrics, dict) and int(cycle_id) > 0:
        metrics["_direction_resolved_cycle"] = int(cycle_id)


def _finalize_execution_metrics(
    entry: dict,
    metrics: dict,
    dl_dir: TradeDirection,
    prob: float,
    predicted_edge: float,
    *,
    meta_applied: bool,
    score: float,
    symbol: str | None,
    orch: Any | None = None,
    force: bool = False,
) -> tuple[TradeDirection, dict]:
    """Aplica telemetria meta e marca candidato pronto sem veto de sinal."""
    if symbol is not None:
        attach_live_signal_metrics(orch, symbol, metrics)
    apply_live_calib_drift_soft(metrics, orch=orch, symbol=symbol)
    exec_dir, _final_score = apply_meta_regression_edge(
        dl_dir, metrics, predicted_edge, meta_applied=meta_applied, base_score=score, symbol=symbol
    )
    if force:
        metrics.pop("signal_status", None)
        metrics["force_trade_every_cycle"] = True
    metrics["meta_veto_mode"] = "none"
    metrics.update(
        {
            "exec_direction": exec_dir.name,
            "resolved_direction": exec_dir.name,
            "tcn_score": prob,
            "execution_candidate_ready": True,
        }
    )
    ensure_direction_margin(metrics)
    metrics.pop("quality_guard_reject", None)
    metrics.pop("regime_skip_cycle", None)
    metrics.pop("gate_reason", None)
    sync_entry_metrics(entry, metrics)
    return exec_dir, metrics


def resolve_execution_direction(
    entry: dict,
    *,
    exec_cfg: dict | None = None,
    calibration_cfg: dict | None = None,
    recovery_active: bool = False,
    symbol: str | None = None,
    corr_matrix: dict[tuple[str, str], float] | None = None,
    infra_cfg: dict | None = None,
    peer_entry: dict | None = None,
    cycle_id: int = 0,
    risk_manager: Any | None = None,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
) -> tuple[TradeDirection, dict] | None:
    """Resolve direcao micro fiel ao sinal TCN/DL com telemetria meta-regressor."""
    _ = (calibration_cfg, corr_matrix, recovery_active, peer_entry, risk_manager, skipped_cycles_counter)
    exec_cfg_dict = exec_cfg if isinstance(exec_cfg, dict) else {}
    force = force_trade_every_cycle(exec_cfg_dict)
    active_cycle = int(cycle_id or 0)
    if orch is not None:
        active_cycle = int(getattr(orch, "_active_cycle_id", 0) or active_cycle or 0)
    prior = entry.setdefault("metrics", {})
    if active_cycle > 0 and int(prior.get("_direction_resolved_cycle") or 0) == active_cycle:
        ready_name = str(prior.get("exec_direction") or prior.get("resolved_direction") or "").upper()
        if prior.get("execution_candidate_ready") and ready_name in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
            return TradeDirection[ready_name], prior
    checks = initial_direction_checks(
        entry,
        exec_cfg_dict,
        orch=orch,
        skipped_cycles_counter=skipped_cycles_counter,
    )
    if checks is None:
        _stamp_direction_resolved_cycle(entry, active_cycle)
        return None
    dl_dir, metrics, prob = checks
    score = seed_direction_metrics(metrics, dl_dir=dl_dir, prob=prob)
    predicted_edge, meta_applied = resolve_meta_payoff_edge(
        symbol=symbol,
        metrics=metrics,
        direction=dl_dir,
        tcn_probability=prob,
        _base_score=score,
        config={"infra": infra_cfg} if infra_cfg else None,
    )
    if metrics.get("meta_payoff_edge_zscore") is None and metrics.get("edge_zscore") is None:
        attach_payoff_edge_zscore_metrics(
            metrics, float(metrics.get("predicted_payoff_edge", predicted_edge)), symbol=symbol
        )
    result = _finalize_execution_metrics(
        entry,
        metrics,
        dl_dir,
        prob,
        predicted_edge,
        meta_applied=meta_applied,
        score=score,
        symbol=symbol,
        orch=orch,
        force=force,
    )
    _stamp_direction_resolved_cycle(entry, active_cycle)
    return result

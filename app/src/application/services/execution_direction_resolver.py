"""Motor de direcao com refinamento de payoff continuo pelo meta-regressor LightGBM."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_direction_checks import (
    has_meta_zscore_telemetry as _has_meta_zscore_telemetry,
    infer_dl_direction,
    initial_direction_checks,
    is_technically_blocked,
    reject_on_quality_gate,
    seed_direction_metrics,
    sync_entry_metrics,
)
from src.application.services.execution_direction_meta_edge import (
    _negative_edge_skip,
    _resolve_meta_edge_floor,
    _stamp_direction_resolved_cycle,
)
from src.application.services.execution_direction_persistence import _apply_persistence_guard_skip
from src.application.services.execution_price_zone_gate import (
    align_or_keep_meta_side,
    apply_price_zone_gate,
)
from src.application.services.execution_quality_gate import ensure_direction_margin
from src.application.services.force_trade_mode import force_trade_every_cycle
from src.application.services.live_signal_metrics import apply_live_calib_drift_soft, attach_live_signal_metrics
from src.application.services.meta_classifier_stacking import resolve_meta_payoff_edge
from src.application.services.meta_payoff_regression import apply_meta_regression_edge
from src.application.services.meta_payoff_veto_gate import (
    is_execution_signal_vetoed,
    should_veto_meta_payoff_negative_zscore,
)
from src.application.services.payoff_edge_zscore import attach_payoff_edge_zscore_metrics
from src.application.services.side_equilibrium_gate import resolve_direction_with_side_equilibrium
from src.domain.models.trade import TradeDirection


_RECOVERY_RERESOLVE_GATES = frozenset(
    {
        "meta_shadow_inverted_veto",
        "meta_payoff_negative_zscore_veto",
        "meta_negative_edge",
        "side_imbalance_flip_not_better",
        "side_imbalance_thin_margin_flip",
        "side_imbalance_both_sides",
        "side_imbalance_flip_zone_conflict",
        "side_imbalance_large_n_margin",
    }
)

__all__ = (
    "_apply_persistence_guard_skip",
    "_has_meta_zscore_telemetry",
    "_negative_edge_skip",
    "_resolve_meta_edge_floor",
    "_stamp_direction_resolved_cycle",
    "infer_dl_direction",
    "is_technically_blocked",
    "resolve_execution_direction",
)


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
    risk_manager: Any | None = None,
    orch: Any | None = None,
    force: bool = False,
    exec_cfg: dict | None = None,
    recovery_active: bool = False,
    skipped_cycles_counter: int | None = None,
) -> tuple[TradeDirection, dict] | None:
    """Aplica decisao de execucao final."""
    if symbol is not None:
        attach_live_signal_metrics(orch, symbol, metrics)
    apply_live_calib_drift_soft(metrics, orch=orch, symbol=symbol)
    exec_dir, _final_score = apply_meta_regression_edge(
        dl_dir, metrics, predicted_edge, meta_applied=meta_applied, base_score=score, symbol=symbol
    )
    if _negative_edge_skip(
        metrics,
        predicted_edge,
        force=force,
        meta_applied=meta_applied,
        exec_cfg=exec_cfg,
        skipped_cycles_counter=skipped_cycles_counter,
        orch=orch,
        recovery_active=recovery_active,
        risk_manager=risk_manager,
    ):
        sync_entry_metrics(entry, metrics)
        return None
    hard = should_veto_meta_payoff_negative_zscore(
        metrics,
        direction=exec_dir,
        risk_manager=risk_manager,
        orch=orch,
        recovery_active=recovery_active,
    )
    if (hard or is_execution_signal_vetoed(metrics)) and not force:
        sync_entry_metrics(entry, metrics)
        return None
    if force:
        metrics.pop("signal_status", None)
        metrics["meta_veto_mode"] = "none"
        metrics["force_trade_every_cycle"] = True
    metrics.update(
        {
            "exec_direction": exec_dir.name,
            "resolved_direction": exec_dir.name,
            "tcn_score": prob,
        }
    )
    ensure_direction_margin(metrics)
    zone_reason = apply_price_zone_gate(
        metrics, exec_dir, exec_cfg if isinstance(exec_cfg, dict) else {}, tcn_direction=dl_dir
    )
    if zone_reason is not None and not force:
        metrics["quality_guard_reject"] = True
        metrics["regime_skip_cycle"] = True
        metrics["gate_reason"] = zone_reason
        sync_entry_metrics(entry, metrics)
        return None
    exec_dir = align_or_keep_meta_side(
        exec_dir,
        metrics,
        dl_dir=dl_dir,
        predicted_edge=predicted_edge,
        meta_applied=meta_applied,
    )
    metrics["exec_direction"] = exec_dir.name
    metrics["resolved_direction"] = exec_dir.name
    ensure_direction_margin(metrics)
    chosen = resolve_direction_with_side_equilibrium(
        orch,
        symbol,
        exec_dir,
        metrics,
        recovery_active=recovery_active,
    )
    if chosen is None:
        sync_entry_metrics(entry, metrics)
        return None
    if chosen != exec_dir and meta_applied and float(metrics.get("predicted_payoff_edge", predicted_edge)) > 0.0:
        if bool(metrics.get("side_eq_toxic_escape")):
            metrics["side_eq_escape_edge_kept"] = True
        else:
            metrics["predicted_payoff_edge"] = -abs(float(metrics.get("predicted_payoff_edge", predicted_edge)))
            metrics["side_eq_edge_inverted"] = True
    exec_dir = chosen
    if not bool(metrics.get("side_eq_flipped")):
        exec_dir = align_or_keep_meta_side(
            exec_dir,
            metrics,
            dl_dir=dl_dir,
            predicted_edge=float(metrics.get("predicted_payoff_edge", predicted_edge)),
            meta_applied=meta_applied,
        )
    else:
        metrics["price_zone_side_eq_override"] = True
    metrics["exec_direction"] = exec_dir.name
    metrics["resolved_direction"] = exec_dir.name
    ensure_direction_margin(metrics)
    if float(metrics.get("side_eq_margin_boost", 0.0)) > 0.0:
        margin = float(metrics.get("direction_margin", 0.0))
        floor = float(metrics.get("quality_min_direction_margin", 0.0))
        if margin + 1e-12 < floor:
            metrics["gate_reason"] = "side_imbalance_large_n_margin"
            metrics["quality_guard_reject"] = True
            sync_entry_metrics(entry, metrics)
            return None
    if metrics.get("meta_veto_mode") is None:
        metrics["meta_veto_mode"] = "none"
    metrics.pop("quality_guard_reject", None)
    metrics.pop("regime_skip_cycle", None)
    metrics["execution_candidate_ready"] = True
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
    _ = (calibration_cfg, corr_matrix)
    exec_cfg_dict = exec_cfg if isinstance(exec_cfg, dict) else {}
    force = force_trade_every_cycle(exec_cfg_dict)
    active_cycle = int(cycle_id or 0)
    if orch is not None:
        active_cycle = int(getattr(orch, "_active_cycle_id", 0) or active_cycle or 0)
    prior = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    if not force and active_cycle > 0 and int(prior.get("_direction_resolved_cycle") or 0) == active_cycle:
        gate = str(prior.get("gate_reason") or "")
        blocked = bool(prior.get("quality_guard_reject") or gate)
        if blocked and not (recovery_active and gate in _RECOVERY_RERESOLVE_GATES):
            return None
        ready_name = str(prior.get("exec_direction") or prior.get("resolved_direction") or "").upper()
        if prior.get("execution_candidate_ready") and ready_name in {
            TradeDirection.CALL.name,
            TradeDirection.PUT.name,
        }:
            return TradeDirection[ready_name], prior
        if blocked and recovery_active and gate in _RECOVERY_RERESOLVE_GATES:
            if bool(prior.get("_recovery_reresolve_done") or prior.get("_resolved_under_recovery")):
                return None
            prior["_recovery_reresolve_done"] = True
            prior.pop("_direction_resolved_cycle", None)
            prior.pop("quality_guard_reject", None)
            prior.pop("gate_reason", None)
            prior.pop("regime_skip_cycle", None)
    checks = initial_direction_checks(entry, exec_cfg_dict, orch=orch)
    if checks is None:
        _stamp_direction_resolved_cycle(entry, active_cycle)
        return None
    dl_dir, metrics, prob = checks
    effective_infra = infra_cfg if infra_cfg is not None else exec_cfg_dict
    persisted = _apply_persistence_guard_skip(
        entry,
        metrics,
        dl_dir,
        symbol=symbol,
        peer_entry=peer_entry,
        cycle_id=cycle_id,
        infra_cfg=effective_infra,
        force=force,
    )
    if persisted is None:
        _stamp_direction_resolved_cycle(entry, active_cycle)
        return None
    dl_dir = persisted
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
    gate_probe = dict(metrics)
    gate_probe["predicted_payoff_edge"] = float(predicted_edge)
    gate_probe["meta_classifier_applied"] = bool(meta_applied)
    kw = {
        "risk_manager": risk_manager,
        "recovery_active": recovery_active,
        "skipped_cycles_counter": skipped_cycles_counter,
        "orch": orch,
    }
    if reject_on_quality_gate(entry, metrics, gate_probe, exec_cfg_dict, **kw):
        _stamp_direction_resolved_cycle(entry, active_cycle)
        return None
    if bool(exec_cfg_dict.get("require_meta_for_execution", False)) and not meta_applied:
        metrics["gate_reason"] = "meta_unavailable"
        metrics["quality_guard_reject"] = True
        sync_entry_metrics(entry, metrics)
        _stamp_direction_resolved_cycle(entry, active_cycle)
        return None
    result = _finalize_execution_metrics(
        entry,
        metrics,
        dl_dir,
        prob,
        predicted_edge,
        meta_applied=meta_applied,
        score=score,
        symbol=symbol,
        risk_manager=risk_manager,
        orch=orch,
        force=force,
        exec_cfg=exec_cfg_dict,
        recovery_active=recovery_active,
        skipped_cycles_counter=skipped_cycles_counter,
    )
    _stamp_direction_resolved_cycle(entry, active_cycle)
    return result

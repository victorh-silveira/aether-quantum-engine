"""Motor de direcao TCN com telemetria meta e catalogo minimo de SKIP de sinal."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_anti_loss import apply_anti_loss_seed_discord
from src.application.services.execution_direction_checks import (
    infer_dl_direction,
    initial_direction_checks,
    is_technically_blocked,
    seed_direction_metrics,
    sync_entry_metrics,
)
from src.application.services.execution_direction_fusion import apply_direction_fusion, parse_direction_fusion_config
from src.application.services.execution_invert_side import apply_invert_exec_side
from src.application.services.execution_micro_protect import apply_micro_protect_gates
from src.application.services.execution_neg_edge import apply_negative_cal_edge_pause
from src.application.services.execution_quality_gate_margin import ensure_direction_margin, sync_direction_margin
from src.application.services.execution_regime_chop import apply_regime_chop_pause
from src.application.services.execution_regime_gate import apply_regime_boolean_gate
from src.application.services.execution_scale_adapt import apply_scale_direction_adapt, apply_scale_kelly_side_sync
from src.application.services.execution_scale_sizing import apply_scale_kelly_sizing
from src.application.services.execution_scale_vision import compute_scale_directions, format_scale_audit_line
from src.application.services.execution_side_eq_sizing import apply_side_eq_kelly_sizing
from src.application.services.execution_signal_skip import apply_signal_skip_gates
from src.application.services.force_trade_mode import force_trade_every_cycle
from src.application.services.live_signal_metrics import apply_live_calib_drift_soft, attach_live_signal_metrics
from src.application.services.loss_classifier_gate import apply_loss_classifier_gate
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
    """Aplica telemetria meta, SCALE e catalogo minimo de SKIP de sinal."""
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
    metrics["tcn_direction"] = dl_dir.name
    metrics.update(
        {
            "exec_direction": exec_dir.name,
            "resolved_direction": exec_dir.name,
            "tcn_score": prob,
            "execution_candidate_ready": True,
        }
    )
    ensure_direction_margin(metrics)
    if orch is not None:
        risk_manager = getattr(orch, "risk_manager", None)
        total_fn = getattr(risk_manager, "pending_loss_total", None) if risk_manager is not None else None
        if callable(total_fn):
            try:
                metrics["pending_loss_total"] = max(0.0, float(total_fn()))
            except (TypeError, ValueError):
                metrics.setdefault("pending_loss_total", 0.0)
        elif risk_manager is not None and isinstance(getattr(risk_manager, "pending_loss", None), dict):
            try:
                metrics["pending_loss_total"] = max(0.0, float(sum(risk_manager.pending_loss.values())))
            except (TypeError, ValueError):
                metrics.setdefault("pending_loss_total", 0.0)
    compute_scale_directions(orch, symbol, exec_dir, metrics)
    fusion_raw = None
    if orch is not None and isinstance(getattr(orch, "config", None), dict):
        orch_block = orch.config.get("orchestrator")
        if isinstance(orch_block, dict):
            ex_block = orch_block.get("execution")
            if isinstance(ex_block, dict) and isinstance(ex_block.get("scale_vision"), dict):
                fusion_raw = ex_block["scale_vision"]
    fusion_cfg = parse_direction_fusion_config(fusion_raw)
    replace_adapt = bool(fusion_cfg.get("fusion_enabled")) and bool(fusion_cfg.get("fusion_replace_adapt_flip"))
    if not replace_adapt:
        exec_dir = apply_scale_direction_adapt(metrics, exec_dir)
    else:
        metrics.setdefault("scale_adapted", False)
        metrics.setdefault("scale_adapt_reason", "fusion_replace")
    metrics["exec_direction"] = exec_dir.name
    metrics["resolved_direction"] = exec_dir.name
    metrics["execution_candidate_ready"] = True
    apply_scale_kelly_side_sync(metrics, exec_dir)
    sync_direction_margin(metrics, direction=exec_dir.name)
    apply_side_eq_kelly_sizing(orch, symbol, exec_dir, metrics)
    apply_scale_kelly_sizing(orch, symbol, exec_dir, metrics)
    metrics["scale_audit"] = format_scale_audit_line(metrics)
    metrics.pop("quality_guard_reject", None)
    metrics.pop("regime_skip_cycle", None)
    metrics.pop("gate_reason", None)
    if orch is not None:
        apply_signal_skip_gates(metrics, exec_dir, orch=orch, force=force, symbol=symbol)
        exec_dir = apply_direction_fusion(metrics, exec_dir, orch=orch, cfg=fusion_cfg)
        apply_scale_kelly_side_sync(metrics, exec_dir)
        sync_direction_margin(metrics, direction=exec_dir.name)
        tcn_ref = TradeDirection[str(metrics.get("tcn_direction") or dl_dir.name).upper()]
        apply_loss_classifier_gate(metrics, tcn_ref, orch=orch, force=force, symbol=symbol)
        apply_anti_loss_seed_discord(metrics, orch=orch, force=force, symbol=symbol)
        ready_name = str(metrics.get("exec_direction") or exec_dir.name).upper()
        if ready_name in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
            exec_dir = TradeDirection[ready_name]
        apply_scale_kelly_side_sync(metrics, exec_dir)
        sync_direction_margin(metrics, direction=exec_dir.name)
        apply_micro_protect_gates(metrics, orch=orch, force=force)
    elif bool(fusion_cfg.get("fusion_enabled")):
        exec_dir = apply_direction_fusion(metrics, exec_dir, orch=orch, cfg=fusion_cfg)
        apply_scale_kelly_side_sync(metrics, exec_dir)
        sync_direction_margin(metrics, direction=exec_dir.name)
    apply_regime_boolean_gate(metrics, orch=orch, force=force)
    apply_regime_chop_pause(metrics, orch=orch, force=force)
    apply_negative_cal_edge_pause(metrics, orch=orch, force=force)
    ready_name = str(metrics.get("exec_direction") or exec_dir.name).upper()
    if ready_name in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
        exec_dir = TradeDirection[ready_name]
    exec_dir = apply_invert_exec_side(metrics, exec_dir, orch=orch)
    apply_scale_kelly_side_sync(metrics, exec_dir)
    sync_direction_margin(metrics, direction=exec_dir.name)
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

"""Motor de direcao: TCN + FLIP por p_eff do loss-clf; SKIP tecnico e neg_edge."""

from __future__ import annotations

from typing import Any

from src.application.services.deep_learning.dl_gating import resolve_side_edge
from src.application.services.direction_error_reversal import apply_error_reversal_to_direction
from src.application.services.direction_loss_tracker import should_anti_trend_lock_flip
from src.application.services.execution_direction_checks import (
    apply_invert_exec_side,
    infer_dl_direction,
    initial_direction_checks,
    is_technically_blocked,
    seed_direction_metrics,
    stamp_direction_resolved_cycle,
    sync_entry_metrics,
    sync_kelly_side,
)
from src.application.services.execution_four_vetoes import reevaluate_market_direction
from src.application.services.execution_payout import resolve_execution_payout
from src.application.services.execution_quality_gate_margin import ensure_direction_margin, sync_direction_margin
from src.application.services.execution_scale_vision import compute_scale_directions
from src.application.services.execution_senior_skips import apply_senior_execution_skips
from src.application.services.execution_side_eq_sizing import apply_side_eq_kelly_sizing
from src.application.services.execution_signal_skips import should_skip_acc_floor, should_skip_neg_edge
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
    exec_cfg: dict | None = None,
) -> tuple[TradeDirection, dict] | None:
    """TCN → FLIP loss-clf → invert → SKIP neg_edge → Kelly."""
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
    payout = resolve_execution_payout(orch)
    metrics["payout_assumed"] = payout
    if orch is not None:
        rm = getattr(orch, "risk_manager", None)
        tot_fn, pend_dict = getattr(rm, "pending_loss_total", None), getattr(rm, "pending_loss", None)
        try:
            val = (
                float(tot_fn())
                if callable(tot_fn)
                else (sum(pend_dict.values()) if isinstance(pend_dict, dict) else 0.0)
            )
            metrics["pending_loss_total"] = max(0.0, float(val))
        except (TypeError, ValueError):
            metrics.setdefault("pending_loss_total", 0.0)
    for k in ("quality_guard_reject", "regime_skip_cycle", "gate_reason"):
        metrics.pop(k, None)
    if orch is not None:
        tcn_ref = TradeDirection[str(metrics.get("tcn_direction") or dl_dir.name).upper()]
        apply_loss_classifier_gate(metrics, tcn_ref, orch=orch, force=force, symbol=symbol)
        ready_name = str(metrics.get("exec_direction") or exec_dir.name).upper()
        if ready_name in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
            exec_dir = TradeDirection[ready_name]
    if orch is not None and symbol:
        compute_scale_directions(orch, str(symbol), exec_dir, metrics)
    allow_flip = bool((exec_cfg or {}).get("allow_direction_flip", True))
    anti_trend_active = bool((exec_cfg or {}).get("anti_trend_lock", (exec_cfg or {}).get("enable_anti_trend", True)))
    if allow_flip and anti_trend_active and not bool(metrics.get("loss_clf_flip")):
        pend = float(metrics.get("pending_loss_total", 0.0) or 0.0)
        trend = str(metrics.get("trend_direction") or "").strip().upper()
        curr_edge = float(metrics.get("cal_side_edge", predicted_edge) or 0.0)
        zeta = float(metrics.get("elastic_distance_ou", 0.0) or 0.0)
        micro_reg = str(metrics.get("scale_micro_regime") or "").strip().lower()
        reg_side = str(metrics.get("scale_micro_side") or "").strip().upper()
        veto_weak = bool((exec_cfg or {}).get("veto_weak_regime_flip", False))
        closed_cd = str(metrics.get("closed_micro_candle_dir") or "").strip().upper()
        ind_map = metrics.get("indicators") if isinstance(metrics.get("indicators"), dict) else {}
        if should_anti_trend_lock_flip(
            symbol,
            exec_dir,
            pending_loss_total=pend,
            edge=curr_edge,
            prob=prob,
            trend_direction=trend,
            elastic_zeta=zeta,
            micro_regime=micro_reg,
            regime_side=reg_side,
            closed_candle=closed_cd,
            rsi=ind_map.get("rsi"),
        ):
            flipped = TradeDirection.PUT if exec_dir == TradeDirection.CALL else TradeDirection.CALL
            metrics["anti_trend_lock_flip"] = True
            metrics["anti_trend_lock_from"], metrics["anti_trend_lock_to"] = exec_dir.name, flipped.name
            if (exec_dir == TradeDirection.CALL and zeta > 2.0) or (exec_dir == TradeDirection.PUT and zeta < -2.0):
                metrics["anti_trend_lock_reason"] = "OU_ELASTIC_EXHAUSTION"
            elif micro_reg in {"explosion", "retraction"} and reg_side in {"CALL", "PUT"}:
                metrics["anti_trend_lock_reason"] = f"COUNTER_{micro_reg.upper()}_ALIGNMENT"
            else:
                metrics["anti_trend_lock_reason"] = "CONSECUTIVE_DIRECTION_LOSSES"
            cal_p = float(metrics.get("calibrated_prob") or 0.5)
            p_flip = cal_p if flipped == TradeDirection.CALL else 1.0 - cal_p
            trend_aligned = trend in {TradeDirection.CALL.name, TradeDirection.PUT.name} and flipped.name == trend
            if veto_weak and not trend_aligned and p_flip + 1e-9 < 0.50:
                metrics["anti_trend_lock_flip"] = False
                metrics.pop("anti_trend_lock_from", None)
                metrics.pop("anti_trend_lock_to", None)
                metrics.pop("anti_trend_lock_reason", None)
            else:
                exec_dir = flipped
    exec_dir, _ = apply_error_reversal_to_direction(orch, str(symbol or ""), exec_dir, metrics, exec_cfg=exec_cfg)
    if bool(metrics.get("alpha_flip_applied")):
        metrics["direction_origin"] = "FLIP_ERROR_DRIVEN_ALPHA"
    elif bool(metrics.get("loss_clf_flip")):
        metrics["direction_origin"] = "FLIP_LOSS_CLF"
    elif bool(metrics.get("anti_trend_lock_flip")):
        metrics["direction_origin"] = "FLIP_ANTI_TREND_LOCK"
    else:
        metrics["direction_origin"] = "TCN_DIRECT"
    metrics["exec_direction_pre_scale"] = exec_dir.name
    metrics["scale_adapt_applied"] = False
    metrics.pop("scale_adapt_reason", None)
    exec_dir = apply_invert_exec_side(exec_dir, metrics, exec_cfg)
    exec_dir = reevaluate_market_direction(exec_dir, metrics, exec_cfg, payout=payout, orch=orch, symbol=symbol)
    metrics["exec_direction"] = exec_dir.name
    metrics["resolved_direction"] = exec_dir.name
    cal_prob = metrics.get("calibrated_prob")
    if cal_prob is not None:
        try:
            if bool(metrics.get("loss_clf_flip")):
                p_eff = float(metrics.get("loss_clf_p_eff") or metrics.get("loss_clf_p_loss") or 0.58)
                metrics["cal_side_edge"] = float((p_eff * (1.0 + payout)) - 1.0)
            elif bool(metrics.get("anti_trend_lock_flip")):
                cal_p = float(metrics.get("calibrated_prob") or 0.5)
                p_dir = cal_p if exec_dir == TradeDirection.CALL else 1.0 - cal_p
                t_val = str(metrics.get("trend_direction") or "").strip().upper()
                if t_val in {TradeDirection.CALL.name, TradeDirection.PUT.name} and exec_dir.name == t_val:
                    metrics["cal_side_edge"] = float((0.58 * (1.0 + payout)) - 1.0)
                    metrics["conviction"] = 0.58
                    metrics["trade_score"] = 0.58
                else:
                    metrics["cal_side_edge"] = float((p_dir * (1.0 + payout)) - 1.0)
                    metrics["conviction"] = p_dir
                    metrics["trade_score"] = p_dir
            elif bool(metrics.get("alpha_flip_applied")):
                cal_p = float(metrics.get("calibrated_prob") or 0.5)
                p_dir = cal_p if exec_dir == TradeDirection.CALL else 1.0 - cal_p
                metrics["cal_side_edge"] = float((p_dir * (1.0 + payout)) - 1.0)
            else:
                metrics["cal_side_edge"] = resolve_side_edge(
                    float(cal_prob),
                    direction=exec_dir,
                    payout=payout,
                )
        except (TypeError, ValueError):
            pass
    if should_skip_neg_edge(metrics, exec_cfg, force=force):
        sync_entry_metrics(entry, metrics)
        return None
    exec_dir, blocked = apply_senior_execution_skips(
        exec_dir,
        metrics,
        exec_cfg=exec_cfg,
        orch=orch,
        symbol=symbol,
        force=force,
    )
    if blocked:
        sync_entry_metrics(entry, metrics)
        return None
    sync_kelly_side(metrics, exec_dir)
    sync_direction_margin(metrics, direction=exec_dir.name)
    apply_side_eq_kelly_sizing(orch, symbol, exec_dir, metrics)
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
    """Resolve direcao: TCN + FLIP loss-clf; telemetria meta sem soft Kelly."""
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
        stamp_direction_resolved_cycle(entry, active_cycle)
        return None
    dl_dir, metrics, prob = checks
    if should_skip_acc_floor(metrics, exec_cfg_dict, orch=orch, force=force):
        sync_entry_metrics(entry, metrics)
        stamp_direction_resolved_cycle(entry, active_cycle)
        return None
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
        exec_cfg=exec_cfg_dict,
    )
    stamp_direction_resolved_cycle(entry, active_cycle)
    return result

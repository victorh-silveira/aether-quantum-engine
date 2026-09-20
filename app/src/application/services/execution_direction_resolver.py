"""Motor de direcao: TCN + FLIP por p_eff do loss-clf; SKIP tecnico e neg_edge."""

from __future__ import annotations

from typing import Any

from src.application.services.deep_learning.dl_gating import MARKET_PAYOUT_SSOT, resolve_side_edge
from src.application.services.direction_loss_tracker import should_anti_trend_lock_flip
from src.application.services.execution_direction_checks import (
    infer_dl_direction,
    initial_direction_checks,
    is_technically_blocked,
    seed_direction_metrics,
    sync_entry_metrics,
)
from src.application.services.execution_market_confluence import (
    should_skip_chop_congestion,
    should_skip_directional_momentum_discord,
    should_skip_exhaustion,
    should_skip_two_bar_momentum_trap,
)
from src.application.services.execution_price_action import (
    should_skip_adverse_tick_flow,
    should_skip_climactic_blowoff,
    should_skip_opposing_marubozu_flow,
    should_skip_wick_rejection,
)
from src.application.services.execution_quality_gate_margin import ensure_direction_margin, sync_direction_margin
from src.application.services.execution_scale_vision import compute_scale_directions
from src.application.services.execution_senior_confluence import evaluate_senior_directional_decision
from src.application.services.execution_side_eq_sizing import apply_side_eq_kelly_sizing
from src.application.services.execution_signal_skips import (
    should_skip_acc_floor,
    should_skip_neg_edge,
    should_skip_trend_discord,
)
from src.application.services.force_trade_mode import force_trade_every_cycle
from src.application.services.live_signal_metrics import apply_live_calib_drift_soft, attach_live_signal_metrics
from src.application.services.loss_classifier_gate import apply_loss_classifier_gate
from src.application.services.meta_classifier_stacking import resolve_meta_payoff_edge
from src.application.services.meta_payoff_regression import apply_meta_regression_edge
from src.application.services.payoff_edge_zscore import attach_payoff_edge_zscore_metrics
from src.domain.models.trade import TradeDirection
from src.domain.risk.kelly_p_align import apply_kelly_side_p
from src.domain.risk.kelly_runtime_config import load_kelly_runtime_from_settings


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


def _sync_kelly_side(metrics: dict[str, Any], exec_dir: TradeDirection) -> None:
    """Alinha p Kelly ao lado EXEC com piso SSOT."""
    rt = load_kelly_runtime_from_settings()
    conviction = float(metrics.get("conviction", metrics.get("trade_score", 0.5)) or 0.5)
    apply_kelly_side_p(
        metrics,
        order_direction=exec_dir.name,
        kelly_config={"kelly_p_floor": rt["kelly_p_floor"]},
        conviction=conviction,
    )


def _apply_invert_exec_side(
    exec_dir: TradeDirection,
    metrics: dict[str, Any],
    exec_cfg: dict | None,
) -> TradeDirection:
    """Inverte CALL↔PUT se invert_exec_side=true."""
    enabled = bool((exec_cfg or {}).get("invert_exec_side", False))
    metrics["invert_exec_side"] = enabled
    if not enabled:
        return exec_dir
    metrics["exec_direction_pre_invert"] = exec_dir.name
    if exec_dir == TradeDirection.CALL:
        return TradeDirection.PUT
    return TradeDirection.CALL


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
    if orch is not None:
        rm = getattr(orch, "risk_manager", None)
        tot_fn = getattr(rm, "pending_loss_total", None)
        pend_dict = getattr(rm, "pending_loss", None)
        try:
            val = (
                float(tot_fn()) if callable(tot_fn) else sum(pend_dict.values()) if isinstance(pend_dict, dict) else 0.0
            )
            metrics["pending_loss_total"] = max(0.0, float(val))
        except (TypeError, ValueError):
            metrics.setdefault("pending_loss_total", 0.0)
    metrics.pop("quality_guard_reject", None)
    metrics.pop("regime_skip_cycle", None)
    metrics.pop("gate_reason", None)
    if orch is not None:
        tcn_ref = TradeDirection[str(metrics.get("tcn_direction") or dl_dir.name).upper()]
        apply_loss_classifier_gate(metrics, tcn_ref, orch=orch, force=force, symbol=symbol)
        ready_name = str(metrics.get("exec_direction") or exec_dir.name).upper()
        if ready_name in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
            exec_dir = TradeDirection[ready_name]
    if not bool(metrics.get("loss_clf_flip")):
        pend = float(metrics.get("pending_loss_total", 0.0) or 0.0)
        trend = str(metrics.get("trend_direction") or "").strip().upper()
        curr_edge = float(metrics.get("cal_side_edge", predicted_edge) or 0.0)
        if should_anti_trend_lock_flip(
            symbol, exec_dir, pending_loss_total=pend, edge=curr_edge, prob=prob, trend_direction=trend
        ):
            flipped = TradeDirection.PUT if exec_dir == TradeDirection.CALL else TradeDirection.CALL
            metrics["anti_trend_lock_flip"] = True
            metrics["anti_trend_lock_from"] = exec_dir.name
            metrics["anti_trend_lock_to"] = flipped.name
            exec_dir = flipped
    if bool(metrics.get("loss_clf_flip")):
        metrics["direction_origin"] = "FLIP_LOSS_CLF"
    elif bool(metrics.get("anti_trend_lock_flip")):
        metrics["direction_origin"] = "FLIP_ANTI_TREND_LOCK"
    else:
        new_dir, did_flip, reason = evaluate_senior_directional_decision(
            exec_dir, metrics, orch=orch, symbol=symbol, exec_cfg=exec_cfg
        )
        if did_flip:
            metrics["senior_trader_flip"] = True
            metrics["senior_confluence_reason"] = reason
            metrics["senior_flip_from"] = exec_dir.name
            metrics["senior_flip_to"] = new_dir.name
            metrics["direction_origin"] = "FLIP_SENIOR_CONFLUENCE"
            exec_dir = new_dir
        else:
            metrics["direction_origin"] = "TCN_DIRECT"
    metrics["exec_direction_pre_scale"] = exec_dir.name
    metrics["scale_adapt_applied"] = False
    metrics.pop("scale_adapt_reason", None)
    if orch is not None and symbol:
        compute_scale_directions(orch, str(symbol), exec_dir, metrics)
    exec_dir = _apply_invert_exec_side(exec_dir, metrics, exec_cfg)
    metrics["exec_direction"] = exec_dir.name
    metrics["resolved_direction"] = exec_dir.name
    cal_prob = metrics.get("calibrated_prob")
    if cal_prob is not None:
        try:
            if bool(metrics.get("loss_clf_flip")):
                p_eff = float(metrics.get("loss_clf_p_eff") or metrics.get("loss_clf_p_loss") or 0.58)
                metrics["cal_side_edge"] = float((p_eff * (1.0 + MARKET_PAYOUT_SSOT)) - 1.0)
            elif bool(metrics.get("anti_trend_lock_flip")):
                conv = float(metrics.get("conviction") or metrics.get("trade_score") or 0.58)
                p_dir = max(0.55, conv)
                metrics["cal_side_edge"] = float((p_dir * (1.0 + MARKET_PAYOUT_SSOT)) - 1.0)
            elif bool(metrics.get("senior_trader_flip")):
                p_dir = max(0.56, float(metrics.get("conviction") or 0.56))
                metrics["calibrated_prob"] = p_dir
                metrics["conviction"] = p_dir
                metrics["cal_side_edge"] = float((p_dir * (1.0 + MARKET_PAYOUT_SSOT)) - 1.0)
            else:
                metrics["cal_side_edge"] = resolve_side_edge(
                    float(cal_prob),
                    direction=exec_dir,
                    payout=MARKET_PAYOUT_SSOT,
                )
        except (TypeError, ValueError):
            pass
    for skip_fn in (
        lambda: should_skip_neg_edge(metrics, exec_cfg, force=force),
        lambda: should_skip_trend_discord(metrics, exec_dir, exec_cfg, force=force),
        lambda: should_skip_exhaustion(metrics, exec_dir, exec_cfg, force=force),
        lambda: should_skip_chop_congestion(metrics, exec_cfg, force=force),
        lambda: should_skip_directional_momentum_discord(metrics, exec_dir, exec_cfg, force=force),
        lambda: should_skip_two_bar_momentum_trap(metrics, exec_dir, exec_cfg, force=force),
        lambda: should_skip_wick_rejection(metrics, exec_dir, exec_cfg, orch=orch, symbol=symbol, force=force),
        lambda: should_skip_climactic_blowoff(metrics, exec_dir, exec_cfg, orch=orch, symbol=symbol, force=force),
        lambda: should_skip_opposing_marubozu_flow(metrics, exec_dir, exec_cfg, orch=orch, symbol=symbol, force=force),
        lambda: should_skip_adverse_tick_flow(metrics, exec_dir, exec_cfg, force=force),
    ):
        if skip_fn():
            sync_entry_metrics(entry, metrics)
            return None
    _sync_kelly_side(metrics, exec_dir)
    sync_direction_margin(metrics, direction=exec_dir.name)
    apply_side_eq_kelly_sizing(orch, symbol, exec_dir, metrics)
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
        _stamp_direction_resolved_cycle(entry, active_cycle)
        return None
    dl_dir, metrics, prob = checks
    if should_skip_acc_floor(metrics, exec_cfg_dict, orch=orch, force=force):
        sync_entry_metrics(entry, metrics)
        _stamp_direction_resolved_cycle(entry, active_cycle)
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
    _stamp_direction_resolved_cycle(entry, active_cycle)
    return result

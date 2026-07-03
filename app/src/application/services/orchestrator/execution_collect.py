"""Orchestrator service for collecting and selecting cluster execution candidates."""

__all__ = ["collect_cluster_orders", "_mandatory_fallback_candidates"]

from src.application.services.execution_direction import build_execution_candidate
from src.application.services.execution_quality_asymmetric_gate import (
    validate_micro_boundary_saturation_gate,
    validate_micro_noise_gate,
    validate_recovery_asymmetric_gate,
)
from src.application.services.execution_symbols import (
    select_best_execution_candidate,
    select_mandatory_execution_candidate,
)
from src.application.services.execution_symbols_recovery import (
    apply_recovery_direction_flip,
    pending_recovery_active,
)
from src.application.services.execution_universal_regime_gate import regime_skip_blocks_trade
from src.application.services.orchestrator.execution_collect_gather import gather_cluster_candidates
from src.application.services.orchestrator.execution_collect_helpers import (
    apply_recovery_hedge_to_candidates,
    extract_collect_params,
    log_execution_decision,
    mandatory_fallback_candidates as _mandatory_fallback_candidates,
    mandatory_fallback_if_empty,
    resolve_mandatory_ultimate_candidate,
)
from src.domain.models.trade import TradeDirection
from src.domain.risk.recovery_hurst_decay import session_drawdown_from_profit
from src.domain.risk.stake_sizing import enrich_metrics_conviction, raw_side_from_metrics


_SKIP_LABELS = {
    "low_conviction_neutral_skip": "LOW CONVICTION NEUTRAL SKIP",
    "micro_adx_chop_skip": "MICRO ADX CHOP SKIP",
    "micro_squeeze_breakout_skip": "MICRO SQUEEZE BREAKOUT SKIP",
    "micro_boundary_saturation_skip": "MICRO BOUNDARY SATURATION SKIP",
    "micro_middle_uncertainty_skip": "MICRO MIDDLE UNCERTAINTY SKIP",
}


def _skip_label_for_gate(gate_reason, default_label: str) -> str:
    """Traduz gate_reason em rotulo humano de skip para auditoria de ciclo."""
    return _SKIP_LABELS.get(str(gate_reason or ""), default_label)


def _regime_skip_blocks_mandatory_cycle(exec_mgr, decisions: dict, *, recovery_active: bool) -> tuple[bool, str]:
    """True quando todos os simbolos elegiveis estao sob skip de regime macro."""
    exec_cfg = exec_mgr.orch.config.get("orchestrator", {}).get("execution", {})
    calibration_cfg = exec_mgr.orch.config.get("deep_learning", {}).get("calibration")
    saw_decision = False
    saw_tradeable = False
    skip_label = "REGIME MACRO ENTROPIC_NOISE"
    for symbol in exec_mgr._trade_symbols():
        entry = decisions.get(symbol)
        if not entry:
            continue
        saw_decision = True
        built = build_execution_candidate(
            symbol,
            entry,
            exec_cfg=exec_cfg,
            calibration_cfg=calibration_cfg,
            recovery_active=recovery_active,
        )
        if built is None:
            continue
        _, _, metrics = built
        validate_micro_noise_gate(metrics, exec_cfg=exec_cfg)
        validate_recovery_asymmetric_gate(metrics)
        validate_micro_boundary_saturation_gate(metrics)
        if regime_skip_blocks_trade(metrics):
            skip_label = _skip_label_for_gate(metrics.get("gate_reason"), skip_label)
            continue
        saw_tradeable = True
        break
    return saw_decision and not saw_tradeable, skip_label


def _select_cluster_best(exec_mgr, candidates, *, mandatory, last_loss, last_loss_dir, recovery_active, skip_symbols):
    """Escolhe o melhor candidato do cluster para execucao no ciclo."""
    if not candidates:
        return None
    exec_cfg = exec_mgr.orch.config.get("orchestrator", {}).get("execution", {})
    margin = float(exec_cfg.get("diversify_after_loss_margin", 0.08))
    if mandatory:
        return select_mandatory_execution_candidate(
            exec_mgr.orch,
            candidates,
            last_loss_symbol=last_loss,
            last_loss_direction=last_loss_dir,
            diversify_margin=margin,
            recovery_active=recovery_active,
            skip_symbols=skip_symbols,
        )
    return select_best_execution_candidate(
        candidates,
        last_loss_symbol=last_loss,
        last_loss_direction=last_loss_dir,
        diversify_margin=margin,
        recovery_active=recovery_active,
    )


def collect_cluster_orders(exec_mgr, decisions: dict) -> list[tuple[str, TradeDirection, dict]]:
    """Seleciona uma ordem por ciclo em modo continuo obrigatorio."""
    mandatory = exec_mgr._mandatory_trade_each_cycle()
    recovery_active = pending_recovery_active(getattr(exec_mgr.orch.risk_manager, "pending_loss", {}))
    dl_cfg = exec_mgr.orch.config.get("deep_learning", {})
    (
        recovery_cfg,
        kelly_cfg,
        skip_symbols,
        min_signal,
        min_val,
        min_edge,
        last_loss,
        last_loss_dir,
        mean_reversion,
        low_accuracy,
        exec_cfg,
    ) = extract_collect_params(exec_mgr, dl_cfg, recovery_active=recovery_active)
    cid = f"C{int(exec_mgr.orch._active_cycle_id):04d}"
    consecutive = getattr(exec_mgr.orch.risk_manager, "consecutive_losses_linear", 0)
    skip_counter = int(getattr(exec_mgr.orch, "_recovery_skip_counter", 0))
    session_drawdown = session_drawdown_from_profit(getattr(exec_mgr.orch.risk_manager, "total_session_profit", 0.0))

    candidates = gather_cluster_candidates(
        exec_mgr,
        decisions,
        recovery_active=recovery_active,
        recovery_cfg=recovery_cfg,
        cid=cid,
        min_signal=min_signal,
        min_val=min_val,
        min_edge=min_edge,
        kelly_cfg=kelly_cfg,
        consecutive_losses=consecutive,
        recovery_skip_counter=skip_counter,
        session_drawdown=session_drawdown,
    )
    regime_blocked, skip_label = _regime_skip_blocks_mandatory_cycle(
        exec_mgr,
        decisions,
        recovery_active=recovery_active,
    )
    if not candidates and regime_blocked:
        exec_mgr.logger.info("[%s] SKIP: %s | ciclo bloqueado", cid, skip_label)
        return []
    candidates = mandatory_fallback_if_empty(
        exec_mgr,
        decisions,
        candidates,
        mandatory=mandatory,
        recovery_active=recovery_active,
        last_loss=last_loss,
        last_loss_dir=last_loss_dir,
        skip_symbols=skip_symbols,
        min_signal=min_signal,
        min_val=min_val,
        mean_reversion=mean_reversion,
        low_accuracy=low_accuracy,
    )
    if candidates and skip_symbols:
        candidates = [item for item in candidates if item[0] not in skip_symbols]
    if candidates:
        candidates = apply_recovery_hedge_to_candidates(exec_mgr, candidates, decisions, cid=cid, mandatory=mandatory)
        candidates = mandatory_fallback_if_empty(
            exec_mgr,
            decisions,
            candidates,
            mandatory=mandatory,
            recovery_active=recovery_active,
            last_loss=last_loss,
            last_loss_dir=last_loss_dir,
            skip_symbols=skip_symbols,
            min_signal=min_signal,
            min_val=min_val,
            mean_reversion=mean_reversion,
            low_accuracy=low_accuracy,
        )
    best = _select_cluster_best(
        exec_mgr,
        candidates,
        mandatory=mandatory,
        last_loss=last_loss,
        last_loss_dir=last_loss_dir,
        recovery_active=recovery_active,
        skip_symbols=skip_symbols,
    )
    if best is None and mandatory:
        best, candidates = resolve_mandatory_ultimate_candidate(
            exec_mgr,
            decisions,
            mandatory=mandatory,
            recovery_active=recovery_active,
            last_loss=last_loss,
            last_loss_dir=last_loss_dir,
            skip_symbols=skip_symbols,
            min_signal=min_signal,
            min_val=min_val,
            mean_reversion=mean_reversion,
            low_accuracy=low_accuracy,
        )
    best = apply_recovery_direction_flip(
        best,
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss,
        last_loss_direction=last_loss_dir,
        flip_enabled=bool(exec_cfg.get("recovery_flip_direction_after_loss", True)),
        flip_max_conviction=float(exec_cfg.get("recovery_flip_max_conviction", 0.56)),
        consecutive_losses=getattr(exec_mgr.orch.risk_manager, "consecutive_losses_linear", 0),
        flip_use_trend=bool(exec_cfg.get("recovery_flip_use_trend_confirmation", False)),
    )
    if best is not None:
        metrics = best[2]
        min_raw = float(kelly_cfg.get("stake_conviction_min_raw", 0.51))
        enrich_metrics_conviction(metrics, min_raw=min_raw)
        calibrated = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
        raw_side = raw_side_from_metrics(metrics)
        effective_signal = max(calibrated, raw_side)
        log_execution_decision(exec_mgr, cid, best, candidates, effective_signal)
        return [best]
    return []

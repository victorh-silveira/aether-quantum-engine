"""Coleta e ranking de candidatos de execucao do cluster."""

from typing import Any

from src.application.services.execution_loss_protection import (
    filter_loss_protection_candidates,
    filter_recovery_hurst_candidates,
)
from src.application.services.execution_quality_gate_fallback import cluster_quality_gate_blocks_mandatory_fallback
from src.application.services.execution_symbols import (
    select_best_execution_candidate,
    select_mandatory_execution_candidate,
)
from src.application.services.execution_symbols_recovery import pending_recovery_active
from src.application.services.orchestrator.execution_collect_gather import gather_cluster_candidates
from src.application.services.orchestrator.execution_collect_helpers import (
    extract_collect_params,
    log_execution_decision,
    mandatory_fallback_if_empty as _mandatory_fallback_if_empty,
    resolve_mandatory_ultimate_candidate as _resolve_mandatory_ultimate_candidate,
    revive_ready_cluster_candidates,
)
from src.domain.models.trade import TradeDirection
from src.domain.risk.recovery_hurst_decay import session_drawdown_from_profit
from src.domain.risk.risk_recovery_state import cointegration_redirect_armed, select_cointegration_redirect_candidate
from src.domain.risk.stake_sizing import enrich_metrics_conviction, metric_float, raw_side_from_metrics


def apply_cointegration_redirect(
    candidates: list[tuple[str, TradeDirection, dict]], risk_manager: Any
) -> list[tuple[str, TradeDirection, dict]]:
    """Aplica Consensus Cointegration Redirect sob drawdown > 15% do capital vivo."""
    armed = getattr(risk_manager, "cointegration_redirect_active", None)
    if callable(armed):
        if not bool(armed()):
            return candidates
    else:
        initial_bankroll = float(getattr(risk_manager, "initial_bankroll", 100.0) or 100.0)
        pending_total = getattr(risk_manager, "pending_loss_total", None)
        pending_val = float(pending_total()) if callable(pending_total) else 0.0
        if not cointegration_redirect_armed(initial_bankroll, pending_val):
            return candidates
    redirected = select_cointegration_redirect_candidate(candidates)
    return redirected if redirected else candidates


def _quality_blocks_mandatory_fallback(exec_mgr, decisions: dict) -> bool:
    """Encapsula bloqueio de fallback obrigatorio pelo quality gate em recovery."""
    exec_cfg = exec_mgr.orch.config.get("orchestrator", {}).get("execution", {})
    return cluster_quality_gate_blocks_mandatory_fallback(
        decisions,
        exec_cfg=exec_cfg if isinstance(exec_cfg, dict) else {},
        risk_manager=getattr(exec_mgr.orch, "risk_manager", None),
        trade_symbols=exec_mgr._trade_symbols(),
    )


def mandatory_fallback_if_empty(exec_mgr, decisions, candidates, **kwargs):
    """Aplica fallback obrigatorio respeitando veto de qualidade em recovery."""
    mandatory = bool(kwargs.get("mandatory", exec_mgr._mandatory_trade_each_cycle()))
    if candidates or not mandatory:
        return candidates
    if _quality_blocks_mandatory_fallback(exec_mgr, decisions):
        return []
    return _mandatory_fallback_if_empty(exec_mgr, decisions, candidates, **kwargs)


def resolve_mandatory_ultimate_candidate(exec_mgr, decisions, **kwargs):
    """Ultimo recurso de candidato com veto de qualidade em recovery."""
    if _quality_blocks_mandatory_fallback(exec_mgr, decisions):
        return None, None
    return _resolve_mandatory_ultimate_candidate(exec_mgr, decisions, **kwargs)


def _select_cluster_best(exec_mgr, candidates, *, mandatory, last_loss, recovery_active, skip_symbols):
    """Escolhe o melhor candidato do cluster para execucao no ciclo."""
    if not candidates:
        return None
    if mandatory:
        return select_mandatory_execution_candidate(
            exec_mgr.orch,
            candidates,
            last_loss_symbol=last_loss,
            recovery_active=recovery_active,
            skip_symbols=skip_symbols,
        )
    return select_best_execution_candidate(candidates, last_loss_symbol=last_loss, recovery_active=recovery_active)


def collect_cluster_orders(exec_mgr, decisions: dict) -> list[tuple[str, TradeDirection, dict]]:
    """Seleciona uma ordem por ciclo em modo continuo obrigatorio."""
    mandatory = exec_mgr._mandatory_trade_each_cycle()
    rm = exec_mgr.orch.risk_manager
    recovery_active = pending_recovery_active(
        getattr(rm, "pending_loss", {}),
        int(getattr(rm, "consecutive_losses_linear", 0) or 0),
    )
    dl_cfg = exec_mgr.orch.config.get("deep_learning", {})
    (kelly_cfg, skip_symbols, min_signal, min_val, min_edge, last_loss, exec_cfg) = extract_collect_params(
        exec_mgr, dl_cfg, recovery_active=recovery_active
    )
    cid = f"C{int(exec_mgr.orch._active_cycle_id):04d}"
    consecutive = getattr(exec_mgr.orch.risk_manager, "consecutive_losses_linear", 0)
    skip_counter = int(getattr(exec_mgr.orch, "_recovery_skip_counter", 0))
    session_drawdown = session_drawdown_from_profit(getattr(exec_mgr.orch.risk_manager, "total_session_profit", 0.0))

    candidates = gather_cluster_candidates(
        exec_mgr,
        decisions,
        recovery_active=recovery_active,
        cid=cid,
        min_signal=min_signal,
        min_val=min_val,
        min_edge=min_edge,
        kelly_cfg=kelly_cfg,
        consecutive_losses=consecutive,
        recovery_skip_counter=skip_counter,
        session_drawdown=session_drawdown,
    )
    if not candidates:
        candidates = revive_ready_cluster_candidates(exec_mgr, decisions)
    candidates = filter_loss_protection_candidates(
        candidates, exec_cfg=exec_cfg, recovery_active=recovery_active, consecutive_losses=consecutive
    )
    candidates = filter_recovery_hurst_candidates(
        candidates,
        kelly_cfg=kelly_cfg,
        consecutive_losses=consecutive,
        recovery_skip_counter=skip_counter,
        session_drawdown=session_drawdown,
    )
    candidates = mandatory_fallback_if_empty(
        exec_mgr,
        decisions,
        candidates,
        mandatory=mandatory,
        recovery_active=recovery_active,
        last_loss=last_loss,
        skip_symbols=skip_symbols,
        min_signal=min_signal,
        min_val=min_val,
    )
    pre_skip = list(candidates)
    active_skip = skip_symbols
    if candidates and skip_symbols:
        candidates = [item for item in candidates if item[0] not in skip_symbols]
    if not candidates and pre_skip and recovery_active:
        candidates = pre_skip
        active_skip = frozenset()
    candidates = mandatory_fallback_if_empty(
        exec_mgr,
        decisions,
        candidates,
        mandatory=mandatory,
        recovery_active=recovery_active,
        last_loss=last_loss,
        skip_symbols=active_skip,
        min_signal=min_signal,
        min_val=min_val,
    )
    if not candidates and recovery_active and mandatory and skip_symbols:
        active_skip = frozenset()
        candidates = mandatory_fallback_if_empty(
            exec_mgr,
            decisions,
            candidates,
            mandatory=mandatory,
            recovery_active=recovery_active,
            last_loss=last_loss,
            skip_symbols=active_skip,
            min_signal=min_signal,
            min_val=min_val,
        )
    if candidates:
        candidates = mandatory_fallback_if_empty(
            exec_mgr,
            decisions,
            candidates,
            mandatory=mandatory,
            recovery_active=recovery_active,
            last_loss=last_loss,
            skip_symbols=active_skip,
            min_signal=min_signal,
            min_val=min_val,
        )
    candidates = apply_cointegration_redirect(candidates, exec_mgr.orch.risk_manager)

    best = _select_cluster_best(
        exec_mgr,
        candidates,
        mandatory=mandatory,
        last_loss=last_loss,
        recovery_active=recovery_active,
        skip_symbols=active_skip,
    )
    if best is None and mandatory:
        best, candidates = resolve_mandatory_ultimate_candidate(
            exec_mgr,
            decisions,
            mandatory=mandatory,
            recovery_active=recovery_active,
            last_loss=last_loss,
            skip_symbols=active_skip,
            min_signal=min_signal,
            min_val=min_val,
        )
        if best is None and recovery_active and skip_symbols:
            best, candidates = resolve_mandatory_ultimate_candidate(
                exec_mgr,
                decisions,
                mandatory=mandatory,
                recovery_active=recovery_active,
                last_loss=last_loss,
                skip_symbols=frozenset(),
                min_signal=min_signal,
                min_val=min_val,
            )
    if best is not None:
        metrics = best[2]
        min_raw = float(kelly_cfg.get("stake_conviction_min_raw", 0.51))
        enrich_metrics_conviction(metrics, min_raw=min_raw)
        calibrated = metric_float(metrics, "trade_score", "conviction", default=0.0)
        raw_side = raw_side_from_metrics(metrics)
        effective_signal = max(calibrated, raw_side)
        log_execution_decision(exec_mgr, cid, best, candidates, effective_signal, decisions=decisions)
        return [best]
    return []

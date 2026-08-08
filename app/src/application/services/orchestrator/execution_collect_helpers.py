"""Helper functions for collect_cluster_orders in execution_collect."""

import asyncio

from src.application.services.execution_direction import build_execution_candidate
from src.application.services.execution_direction_fallback import build_mandatory_fallback_candidate
from src.application.services.execution_entropy_fallback import pick_entropy_fallback_candidate
from src.application.services.execution_mandatory_pick import pick_absolute_mandatory_candidate
from src.application.services.execution_symbols_recovery import recovery_blocked_symbols
from src.application.services.force_trade_mode import force_trade_from_orch, synthesize_force_trade_candidate
from src.application.services.market_audit_log import emit_audit_info, format_indicators_audit_line
from src.application.services.orchestrator.execution_recovery_gate import recovery_min_signal, recovery_min_val_accuracy
from src.domain.models.trade import TradeDirection
from src.domain.risk.recovery_hurst_decay import increment_recovery_skip_counter, resolve_effective_hurst_min
from src.domain.risk.recovery_hurst_gate import recovery_pool_has_persistence


def _finalize_force_trade_candidate(exec_mgr, decisions, forced: tuple[str, TradeDirection, dict] | None):
    """Aplica resolve_execution_direction uma vez no candidato force-trade sintetizado."""
    if forced is None:
        return None
    symbol, _direction, _metrics = forced
    entry = decisions.get(symbol) if isinstance(decisions, dict) else None
    if not isinstance(entry, dict):
        return forced
    exec_cfg = exec_mgr.orch.config.get("orchestrator", {}).get("execution", {})
    rebuilt = build_execution_candidate(
        symbol,
        entry,
        exec_cfg=exec_cfg if isinstance(exec_cfg, dict) else {},
        recovery_active=False,
        cycle_id=int(getattr(exec_mgr.orch, "_active_cycle_id", 0) or 0),
        risk_manager=getattr(exec_mgr.orch, "risk_manager", None),
        skipped_cycles_counter=int(getattr(exec_mgr.orch, "_quality_skipped_cycles_counter", 0) or 0),
        orch=exec_mgr.orch,
    )
    return rebuilt if rebuilt is not None else forced


def mandatory_fallback_candidates(
    exec_mgr, decisions, *, recovery_active, last_loss_symbol, skip_symbols, min_signal, min_val
):
    """Monta lista com candidato forcado quando o pool DL fica vazio."""
    cycle_id = int(getattr(exec_mgr.orch, "_active_cycle_id", 0) or 0)
    entropy_pick = pick_entropy_fallback_candidate(
        exec_mgr._trade_symbols(),
        decisions,
        skip_symbols=skip_symbols,
        recovery_active=recovery_active,
        orch=exec_mgr.orch,
        cycle_id=cycle_id,
    )
    if entropy_pick is not None:
        return [entropy_pick]
    fallback = build_mandatory_fallback_candidate(
        exec_mgr._trade_symbols(),
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss_symbol,
        skip_symbols=skip_symbols,
        min_signal=min_signal,
        min_val=min_val,
        consecutive_losses=getattr(exec_mgr.orch.risk_manager, "consecutive_losses_linear", 0),
        skipped_cycles_counter=int(getattr(exec_mgr.orch, "_quality_skipped_cycles_counter", 0) or 0),
        orch=exec_mgr.orch,
    )
    if fallback is not None:
        return [fallback]
    if force_trade_from_orch(exec_mgr.orch):
        forced = synthesize_force_trade_candidate(exec_mgr._trade_symbols(), decisions, orch=exec_mgr.orch)
        finalized = _finalize_force_trade_candidate(exec_mgr, decisions, forced)
        return [finalized] if finalized is not None else []
    return []


def mandatory_fallback_if_empty(
    exec_mgr, decisions, candidates, *, mandatory, recovery_active, last_loss, skip_symbols, min_signal, min_val
):
    """Aplica fallback obrigatorio quando o pool de candidatos DL fica vazio."""
    if candidates or not mandatory:
        return candidates
    return mandatory_fallback_candidates(
        exec_mgr,
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss,
        skip_symbols=skip_symbols,
        min_signal=min_signal,
        min_val=min_val,
    )


def extract_collect_params(exec_mgr, dl_cfg: dict, *, recovery_active: bool) -> tuple:
    """Extrai parametros de configuracao necessarios para collect_cluster_orders."""
    risk_cfg = exec_mgr.orch.config.get("risk_management", {}) if isinstance(exec_mgr.orch.config, dict) else {}
    kelly_cfg = risk_cfg.get("kelly", {}) if isinstance(risk_cfg, dict) else {}
    proposal_skip_fn = getattr(exec_mgr.orch.risk_manager, "proposal_skip_symbols", None)
    proposal_skip = proposal_skip_fn() if callable(proposal_skip_fn) else frozenset()
    recovery_skip = recovery_blocked_symbols(exec_mgr.orch.risk_manager, kelly_cfg)
    skip_symbols = proposal_skip | recovery_skip
    pending_total = sum(float(v) for v in getattr(exec_mgr.orch.risk_manager, "pending_loss", {}).values())
    consecutive_losses = getattr(exec_mgr.orch.risk_manager, "consecutive_losses_linear", 0)
    min_signal = recovery_min_signal(
        kelly_cfg, recovery_active=recovery_active, pending_total=pending_total, consecutive_losses=consecutive_losses
    )
    min_val = (
        recovery_min_val_accuracy(kelly_cfg, consecutive_losses=consecutive_losses)
        if recovery_active
        else float(dl_cfg.get("min_val_accuracy", 0.54))
    )
    min_edge = float(dl_cfg.get("min_edge_execute", 0.0))
    last_loss = getattr(exec_mgr.orch.risk_manager, "last_loss_symbol", None)
    exec_cfg = exec_mgr.orch.config.get("orchestrator", {}).get("execution", {})
    return (kelly_cfg, skip_symbols, min_signal, min_val, min_edge, last_loss, exec_cfg)


def resolve_mandatory_ultimate_candidate(
    exec_mgr, decisions, *, mandatory, recovery_active, last_loss, skip_symbols, min_signal, min_val
):
    """Ultimo recurso de candidato quando modo obrigatorio nao encontrou best."""
    if not mandatory:
        return None, None
    cycle_id = int(getattr(exec_mgr.orch, "_active_cycle_id", 0) or 0)
    skip_counter = int(getattr(exec_mgr.orch, "_quality_skipped_cycles_counter", 0) or 0)
    ultimate = build_mandatory_fallback_candidate(
        exec_mgr._trade_symbols(),
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss,
        skip_symbols=skip_symbols,
        min_signal=min_signal,
        min_val=min_val,
        consecutive_losses=getattr(exec_mgr.orch.risk_manager, "consecutive_losses_linear", 0),
        skipped_cycles_counter=skip_counter,
        orch=exec_mgr.orch,
    )
    if ultimate is None:
        ultimate = pick_absolute_mandatory_candidate(
            exec_mgr._trade_symbols(),
            decisions,
            recovery_active=recovery_active,
            last_loss_symbol=last_loss,
            min_signal=min_signal,
            min_val=min_val,
            orch=exec_mgr.orch,
            cycle_id=cycle_id,
            risk_manager=getattr(exec_mgr.orch, "risk_manager", None),
            skipped_cycles_counter=skip_counter,
        )
        if ultimate is None:
            ultimate = pick_absolute_mandatory_candidate(
                exec_mgr._trade_symbols(),
                decisions,
                recovery_active=recovery_active,
                last_loss_symbol=last_loss,
                min_signal=0.0,
                min_val=0.0,
                orch=exec_mgr.orch,
                cycle_id=cycle_id,
                risk_manager=getattr(exec_mgr.orch, "risk_manager", None),
                skipped_cycles_counter=skip_counter,
            )
    if ultimate is None and force_trade_from_orch(exec_mgr.orch):
        ultimate = _finalize_force_trade_candidate(
            exec_mgr,
            decisions,
            synthesize_force_trade_candidate(exec_mgr._trade_symbols(), decisions, orch=exec_mgr.orch),
        )
    if ultimate is None:
        return None, None
    return ultimate, [ultimate]


def recovery_hurst_blocks_collect(
    exec_mgr,
    candidates,
    *,
    recovery_active: bool,
    consecutive_losses: int,
    kelly_cfg: dict,
    cid: str,
    recovery_skip_counter: int = 0,
    session_drawdown: float = 0.0,
) -> bool:
    """True quando recovery D'Alembert N2+ deve pular o ciclo por falta de Hurst persistente."""
    hurst_min = resolve_effective_hurst_min(
        kelly_cfg, recovery_skip_counter, consecutive_losses=consecutive_losses, session_drawdown=session_drawdown
    )
    if (
        recovery_active
        and consecutive_losses >= 2
        and candidates
        and not recovery_pool_has_persistence(candidates, consecutive_losses=consecutive_losses, hurst_min=hurst_min)
    ):
        schedule_recovery_skip_counter_increment(exec_mgr.orch)
        new_count = recovery_skip_counter + 1
        exec_mgr.logger.info("[%s] SKIP: Recovery sem Hurst persistente (losses=%d)", cid, consecutive_losses)
        exec_mgr.logger.debug("KELLY: recovery_skip_counter=%d effective_hurst_min=%.2f", new_count, hurst_min)
        return True
    return False


def schedule_recovery_skip_counter_increment(orch) -> None:
    """Persiste incremento do contador Hurst de forma assincrona."""
    store = getattr(orch, "state_store", None)

    async def _persist() -> None:
        """Grava contador Hurst no Redis e atualiza cache local."""
        orch._recovery_skip_counter = await increment_recovery_skip_counter(store)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_persist())
    except RuntimeError:
        orch._recovery_skip_counter = int(getattr(orch, "_recovery_skip_counter", 0)) + 1


def log_execution_decision(
    exec_mgr, cid: str, best: tuple, candidates: list, effective_signal: float, *, decisions: dict | None = None
) -> None:
    """Registra linha IND da decisao de execucao do simbolo escolhido."""
    metrics = best[2]
    cycle_digits = cid[1:] if cid.startswith("C") else cid
    try:
        cycle_id = int(cycle_digits)
    except (TypeError, ValueError):
        cycle_id = int(getattr(exec_mgr.orch, "_active_cycle_id", 0))
    _ = (candidates, decisions, effective_signal)
    emit_audit_info(exec_mgr.logger, format_indicators_audit_line(cycle_id, str(best[0]), metrics))


def revive_ready_cluster_candidates(exec_mgr, decisions) -> list[tuple[str, TradeDirection, dict]]:
    """Reconstroi candidatos prontos quando o pool foi esvaziado apos resolve bem-sucedido."""
    revived: list[tuple[str, TradeDirection, dict]] = []
    for symbol in exec_mgr._trade_symbols():
        entry = decisions.get(symbol) if isinstance(decisions, dict) else None
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict) or not metrics.get("execution_candidate_ready"):
            continue
        if metrics.get("quality_guard_reject"):
            continue
        name = str(metrics.get("exec_direction") or metrics.get("resolved_direction") or "").upper()
        if name not in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
            continue
        revived.append((symbol, TradeDirection[name], metrics))
    return revived

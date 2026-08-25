"""Processamento de outcome e sincronizacao de StateManager na liquidacao."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.deep_learning.dl_outcomes import record_symbol_outcome
from src.application.services.direction_loss_tracker import record_direction_outcome
from src.application.services.live_signal_metrics import record_live_signal_outcome
from src.application.services.loss_classifier_vectors import pop_loss_feature_vector
from src.application.services.meta_classifier_vectors import pop_meta_feature_vector
from src.application.services.orchestrator.post_settlement_loss_cooldown import schedule_post_loss_cooldown
from src.application.services.side_equilibrium_store import record_side_equilibrium_outcome
from src.domain.risk.executed_stake_reconciliation import (
    bind_executed_stake_for_contract,
    reconcile_settlement_profit,
    resolve_executed_buy_stake,
)
from src.domain.risk.stop_win_target import resolve_stop_win_target
from src.infrastructure.inference.loss_classifier_pool import learn_loss_via_config_sync
from src.infrastructure.inference.meta_classifier_pool import learn_meta_via_config_sync


logger = logging.getLogger("AETH")


def _feed_loss_classifier_learn(orch: Any, symbol: str, *, won: bool, contract_id: int) -> None:
    """Envia sample WIN/LOSS ao loss-classifier (fail-open)."""
    vector = pop_loss_feature_vector(orch, str(symbol), int(contract_id))
    if not isinstance(vector, list) or not vector:
        return
    config = getattr(orch, "config", None)
    if not isinstance(config, dict):
        return
    label = "WIN" if won else "LOSS"
    result = learn_loss_via_config_sync(
        config,
        feature_vector=vector,
        label=label,
        contract_id=str(contract_id),
        symbol=str(symbol),
    )
    if not isinstance(result, dict):
        return
    if result.get("skipped") or result.get("error"):
        logger.warning(
            "LOSS_CLF || LEARN falhou label=%s cid=%s detail=%s",
            label,
            contract_id,
            result.get("error") or "skipped",
        )
        return
    detail = (
        f"label={label} buffer_n={result.get('buffer_n', '-')} "
        f"retrained={1 if result.get('retrained') else 0} "
        f"n_train={result.get('n_train', '-')} "
        f"reason={result.get('retrain_skipped_reason') or result.get('retrain_detail') or '-'} "
        f"detail={result.get('retrain_detail') or '-'}"
    )
    orch._last_loss_clf_learn = detail
    logger.info("LOSS_CLF || LEARN %s", detail)


def _feed_meta_classifier_learn(
    orch: Any,
    symbol: str,
    *,
    profit: float,
    stake: float,
    contract_id: int,
) -> None:
    """Envia sample de payoff realizado ao meta-regressor (fail-open)."""
    vector = pop_meta_feature_vector(orch, str(symbol), int(contract_id))
    if not isinstance(vector, list) or not vector:
        return
    config = getattr(orch, "config", None)
    if not isinstance(config, dict):
        return
    denom = max(float(stake), 1e-9)
    target = float(profit) / denom
    result = learn_meta_via_config_sync(
        config,
        feature_vector=vector,
        target=target,
        contract_id=str(contract_id),
        symbol=str(symbol),
    )
    if not isinstance(result, dict):
        return
    if result.get("skipped") or result.get("error"):
        logger.warning(
            "META || LEARN falhou cid=%s detail=%s",
            contract_id,
            result.get("error") or "skipped",
        )
        return
    detail = (
        f"target={target:+.4f} buffer_n={result.get('buffer_n', '-')} "
        f"retrained={1 if result.get('retrained') else 0} "
        f"detail={result.get('retrain_detail') or '-'}"
    )
    orch._last_meta_clf_learn = detail
    logger.info("META || LEARN %s", detail)


def process_contract_outcome(
    orch: Any,
    c: dict,
    contract: Any,
    c_id: int,
    profit: float,
    *,
    audit_direction: str | None = None,
    audit_raw_prob: float | None = None,
    log_cluster_summary: Any,
) -> None:
    """Atualiza o saldo da conta, registra o resultado no risk manager e computa estatisticas da sessao."""
    api_balance = c.get("balance_after")
    orch.state.balance = (
        float(api_balance)
        if api_balance is not None
        and (orch.state.balance <= 0 or abs(float(api_balance) - (orch.state.balance + profit)) <= 2.0)
        else float(orch.state.balance + profit)
    )

    sym = orch.risk_manager.contract_to_symbol.get(c_id, c.get("underlying", "UNK"))
    loss_dir = getattr(contract, "direction", None)
    dir_name = loss_dir.name if loss_dir is not None else None
    if not dir_name and audit_direction:
        dir_name = str(audit_direction)
    planned_stake = float(getattr(contract, "stake", 0.0) or 0.0)
    if planned_stake > 0.0:
        orch.risk_manager.contract_requested_stakes.setdefault(c_id, planned_stake)
    executed_buy = resolve_executed_buy_stake(
        c_id,
        payload=c if isinstance(c, dict) else None,
        contract=contract,
        contract_stakes=orch.risk_manager.contract_stakes,
    )
    profit = reconcile_settlement_profit(profit, executed_buy)
    bind_executed_stake_for_contract(orch.risk_manager.contract_stakes, c_id, executed_buy)
    record_symbol_outcome(orch, sym, won=profit >= 0.0)
    record_live_signal_outcome(
        orch,
        str(sym),
        won=profit >= 0.0,
        raw_prob=audit_raw_prob,
        direction=dir_name or audit_direction,
    )
    if dir_name:
        record_direction_outcome(sym, dir_name, won=profit >= 0.0)
    _feed_loss_classifier_learn(orch, str(sym), won=profit >= 0.0, contract_id=int(c_id))
    _feed_meta_classifier_learn(
        orch,
        str(sym),
        profit=float(profit),
        stake=float(executed_buy),
        contract_id=int(c_id),
    )
    if dir_name and abs(float(profit)) > 1e-12:
        record_side_equilibrium_outcome(
            orch,
            str(sym),
            direction=dir_name,
            won=float(profit) > 0.0,
            profit=float(profit),
            raw_prob=audit_raw_prob,
            cycle_id=int(orch._contract_cycle.get(c_id, 0) or 0),
        )
    orch.risk_manager.register_result(profit, c_id, symbol=sym, current_tick=orch.tick_count, direction=dir_name)
    orch._cluster_results.append({"symbol": sym, "profit": profit})
    orch._last_result_cycle_id = orch._contract_cycle.pop(c_id, 0)
    orch._last_settlement_outcome = "WIN" if profit > 0.0 else ("LOSS" if profit < 0.0 else "FLAT")

    if profit >= 0:
        orch._session_wins += 1
    else:
        orch._session_losses += 1
        orch._last_loss_symbol = sym
        orch._last_loss_direction = dir_name or ""
        schedule_post_loss_cooldown(orch)

    if not orch.risk_manager.active_contract_ids:
        log_cluster_summary(orch)


def sync_state_manager_session(orch: Any, target: float, *, increment_trades: bool) -> bool:
    """Sincroniza StateManager com saldo corrente e retorna se stop-win foi atingido."""
    state_mgr = getattr(orch, "state_mgr", None)
    if state_mgr is None or type(state_mgr).__name__ != "StateManager":
        return False
    if hasattr(state_mgr, "mirror_balance"):
        state_mgr.mirror_balance(float(orch.state.balance))
    else:
        state_mgr.state.current_balance = float(orch.state.balance)
    if state_mgr.state.initial_balance <= 0.0:
        state_mgr.state.initial_balance = float(orch.risk_manager.initial_bankroll)
    if state_mgr.state.daily_stop_win_target <= 0.0:
        state_mgr.state.daily_stop_win_target = float(target)
    if increment_trades:
        state_mgr.state.total_trades_today += 1
    state_mgr.check_session_limits()
    return bool(state_mgr.state.stop_win_triggered)


def check_session_limits_before_post_settlement(orch: Any) -> bool:
    """True quando o stop-win da sessao ja foi atingido antes do ciclo pos-liquidacao."""
    pnl = orch.risk_manager.total_session_profit
    target = resolve_stop_win_target(orch.config.get("risk_management", {}), orch.risk_manager.initial_bankroll)
    if target > 0 and pnl >= target:
        sync_state_manager_session(orch, target, increment_trades=False)
        return True
    if sync_state_manager_session(orch, target, increment_trades=False):
        return True
    state_mgr = getattr(orch, "state_mgr", None)
    if state_mgr is not None and type(state_mgr).__name__ == "StateManager":
        return bool(state_mgr.state.stop_win_triggered)
    return target > 0 and pnl >= target


def update_state_manager_and_check_stop_win(orch: Any, target: float, pnl: float) -> bool:
    """Atualiza o StateManager se disponivel e retorna se o Stop Win foi ativado."""
    triggered = sync_state_manager_session(orch, target, increment_trades=True)
    state_mgr = getattr(orch, "state_mgr", None)
    if state_mgr is not None and type(state_mgr).__name__ == "StateManager":
        state_mgr.save_state()
        return triggered
    return target > 0 and pnl >= target

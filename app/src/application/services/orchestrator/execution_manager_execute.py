"""Execucao sequencial de ordens do cluster com sizing Kelly."""

from __future__ import annotations

import asyncio
from typing import Any

from src.application.services.force_trade_mode import force_trade_from_orch, resolve_force_min_stake
from src.application.services.loss_classifier_vectors import bind_loss_feature_vector_to_contract
from src.application.services.market_audit_log import (
    resolve_meta_payoff_zscore,
    resolve_predicted_edge,
    store_contract_audit,
)
from src.application.services.orchestrator.api_maintenance_guard import handle_broker_maintenance_error
from src.application.services.orchestrator.execution_proposal import is_proposal_runtime_error
from src.domain.models.trade import TradeDirection
from src.domain.risk.stake_sizing import resolve_stake_conviction


async def execute_cluster_orders(
    executor: Any,
    orders: list[tuple[str, TradeDirection, dict]],
    inter_delay: float,
    bankroll_snapshot: float,
) -> int:
    """Executa ordens usando Criterio de Kelly e retorna quantidade enviada."""
    executed_count = 0
    for i, (symbol, direction, metrics) in enumerate(orders):
        order_n = i + 1
        conviction = resolve_stake_conviction(metrics, executor.orch.risk_manager.kelly_config)

        dl_cfg = executor.orch.config.get("deep_learning", {})
        pending = sum(executor.orch.risk_manager.pending_loss.values())
        mandatory = executor._mandatory_trade_each_cycle()
        stop_win_kelly = bool(executor.orch.risk_manager.kelly_config.get("stop_win_kelly_enabled", True))
        stake = executor.orch.risk_manager.calculate_stake(
            bankroll_snapshot,
            symbol,
            conviction=conviction,
            silent=False,
            cycle_id=int(executor.orch._active_cycle_id),
            dl_metrics=metrics,
            order_direction=direction.name,
            max_val_brier=float(dl_cfg.get("max_val_brier_execute", 0.28)),
            mandatory_weak_cap=(
                mandatory and not metrics.get("execute", True) and pending <= 0.0 and not stop_win_kelly
            ),
            mandatory_trade_each_cycle=mandatory,
        )

        if stake <= 0 and force_trade_from_orch(executor.orch):
            stake = resolve_force_min_stake(getattr(executor.orch, "config", None))
        if stake <= 0 and metrics.get("reversal_stake_floor"):
            neutral_pct = float(executor.orch.risk_manager.kelly_config.get("neutral_bankroll_pct", 0.0015))
            _in_recovery = bool(
                int(getattr(executor.orch.risk_manager, "consecutive_losses_linear", 0) or 0) > 0
                or float(
                    executor.orch.risk_manager.pending_loss_total()
                    if callable(getattr(executor.orch.risk_manager, "pending_loss_total", None))
                    else sum(float(v) for v in getattr(executor.orch.risk_manager, "pending_loss", {}).values())
                )
                > 0.0
            )
            pct = max(neutral_pct, 0.008) if _in_recovery else neutral_pct
            stake = max(0.0, bankroll_snapshot * pct)
        if stake <= 0:
            continue

        executor.orch.risk_manager.register_entry_conviction(conviction)

        if i > 0:
            await asyncio.sleep(inter_delay)
        try:
            custom_dur = metrics.get("duration")
            if custom_dur is None:
                risk_params = executor.orch.config.get("risk_management", {}).get("params", {})
                custom_dur = int(risk_params.get("duration", 300))
            order_metrics = {**metrics, "duration": int(custom_dur)}
            res = await executor._place_order(symbol, direction, stake, duration=custom_dur, metrics=order_metrics)
            if res:
                executed_stake = float(getattr(res, "buy_price", 0.0) or stake)
                executor.orch.risk_manager.record_contract_stake(int(res.contract_id), executed_stake)
                executor.orch.risk_manager.active_contract_ids.append(res.contract_id)
                await executor.orch.state.add_contract(res)
                executor.orch._contract_cycle[int(res.contract_id)] = int(executor.orch._active_cycle_id)
                bind_loss_feature_vector_to_contract(executor.orch, str(symbol), int(res.contract_id))
                store_contract_audit(
                    executor.orch,
                    int(res.contract_id),
                    symbol=symbol,
                    direction=direction.name,
                    edge=resolve_predicted_edge(order_metrics),
                    meta_payoff_edge_zscore=resolve_meta_payoff_zscore(order_metrics),
                    raw_prob=(float(order_metrics["raw_prob"]) if order_metrics.get("raw_prob") is not None else None),
                )
                executor._log_exec(
                    symbol,
                    direction,
                    stake,
                    order_metrics,
                    order_n=order_n,
                    contract_id=int(res.contract_id),
                )
                executed_count += 1
        except Exception as e:
            if handle_broker_maintenance_error(executor.orch, e):
                executor.logger.warning(f"SKIP: Sessão fechada para {symbol}: {e}")
                continue
            err_msg = str(e).lower()
            if "closed" in err_msg or "trading is not available" in err_msg:
                executor.logger.warning(f"SKIP: Sessão fechada para {symbol}: {e}")
            else:
                if is_proposal_runtime_error(e):
                    hold = int(
                        executor.orch.config.get("orchestrator", {})
                        .get("execution", {})
                        .get("proposal_failure_skip_cycles", 6)
                    )
                    executor.orch.risk_manager.register_proposal_failure(symbol, cycles=hold)
                executor.logger.error(f"FAIL: EXEC: Falha critica na ordem {symbol}: {e}")
    return executed_count

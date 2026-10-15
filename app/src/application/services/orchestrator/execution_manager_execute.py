"""Execucao sequencial de ordens do cluster com sizing por origem."""

from __future__ import annotations

import asyncio
import math
from typing import Any

from src.application.services.force_trade_mode import force_trade_from_orch, resolve_force_min_stake
from src.application.services.indicator_consensus import (
    available_cluster_bankroll,
    claim_indicator_order_once,
    consensus_enabled,
    quarantine_unknown_purchase,
    release_unbought_order_claim,
    signal_epoch_current,
)
from src.application.services.loss_classifier_vectors import (
    bind_loss_feature_vector_to_contract,
    bind_loss_flip_ctx_to_contract,
)
from src.application.services.market_audit_log import (
    resolve_meta_payoff_zscore,
    resolve_predicted_edge,
    store_contract_audit,
)
from src.application.services.meta_classifier_vectors import (
    bind_meta_feature_vector_to_contract,
    store_meta_feature_vector,
)
from src.application.services.orchestrator.api_maintenance_guard import handle_broker_maintenance_error
from src.application.services.orchestrator.execution_contract_registration import register_confirmed_contract
from src.application.services.orchestrator.execution_model_attribution import record_model_attribution
from src.application.services.orchestrator.execution_proposal import is_proposal_runtime_error
from src.application.services.pullback_recovery_claim import (
    bind_recovery_contract,
    claim_recovery,
    release_recovery,
    unreserved_loss,
)
from src.application.services.rise_fall_quote_guard import resolve_effective_side_probability
from src.domain.models.trade import TradeDirection
from src.domain.risk.checkpoint_stake_cap import checkpoint_stake_cap_pct
from src.domain.risk.pullback_stake import pullback_stake
from src.domain.risk.stake_sizing import resolve_stake_conviction
from src.infrastructure.handlers.trade_handler_errors import PurchaseOutcomeUnknownError


def _discard_opposite_side_training_vectors(orch: Any, symbol: str) -> None:
    """Evita ensinar contratos do indicador com features do lado oposto da TCN."""
    for name in ("_loss_clf_vectors", "_loss_clf_flip_ctx", "_meta_clf_vectors"):
        values = getattr(orch, name, None)
        if isinstance(values, dict):
            values.pop(str(symbol), None)


async def execute_cluster_orders(
    executor: Any,
    orders: list[tuple[str, TradeDirection, dict]],
    inter_delay: float,
    bankroll_snapshot: float,
) -> int:
    """Executa ordens com stake fixa para indicador ou Kelly para TCN."""
    executed_count = 0
    reserved_stake = 0.0
    multi_symbol = consensus_enabled(executor.orch.config) and bool(
        executor.orch.config.get("orchestrator", {}).get("execution", {}).get("multi_symbol_orders", False)
    )
    for i, (symbol, direction, metrics) in enumerate(orders):
        if not signal_epoch_current(executor.orch.config, metrics):
            continue
        order_n = i + 1
        conviction = resolve_stake_conviction(metrics, executor.orch.risk_manager.kelly_config)

        dl_cfg = executor.orch.config.get("deep_learning", {})
        pending = sum(executor.orch.risk_manager.pending_loss.values())
        mandatory = executor._mandatory_trade_each_cycle()
        stop_win_kelly = bool(executor.orch.risk_manager.kelly_config.get("stop_win_kelly_enabled", True))
        available = available_cluster_bankroll(
            bankroll_snapshot,
            reserved_stake,
            float(executor.orch.state.balance) if multi_symbol else bankroll_snapshot,
            multi_symbol=multi_symbol,
        )
        setup = metrics.get("direction_origin") == "PULLBACK_SETUP"
        if setup:
            minimum = float(executor.orch.config.get("risk_management", {}).get("params", {}).get("stake_min", 1.0))
            stake, covered = pullback_stake(
                available, unreserved_loss(executor.orch), float(metrics.get("indicator_quote_rate") or 0.0), minimum
            )
            metrics["_pullback_recovery_amount"] = covered
            metrics["_pullback_stake_cap"] = math.floor(available * 0.05 * 100) / 100
            metrics["indicator_stake_mode"] = "PULLBACK_RECOVERY" if covered > 0 else "PULLBACK_BASE"
        elif metrics.get("indicator_controls_execution"):
            if metrics.get("indicator_demo_pilot"):
                stake = float(executor.orch.config.get("risk_management", {}).get("params", {}).get("stake_min", 1.0))
                metrics["indicator_stake_mode"] = "INDICATOR_DEMO_PILOT"
            else:
                pct = float(executor.orch.config.get("indicator_consensus", {}).get("stake_pct", 0.01))
                pct = max(0.0, min(pct, 0.01))
                stake = math.floor(available * pct * 100) / 100
                metrics["indicator_stake_mode"] = "INDICATOR_FIXED_1PCT"
            if stake > available * 0.01:
                continue
        else:
            stake = executor.orch.risk_manager.calculate_stake(
                available,
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
        if metrics.get("checkpoint_exploration"):
            cap_pct = checkpoint_stake_cap_pct(metrics)
            stake = min(stake, bankroll_snapshot * cap_pct)
            stake_min = float(executor.orch.config.get("risk_management", {}).get("params", {}).get("stake_min", 1.0))
            if stake + 1e-9 < stake_min:
                continue
        if multi_symbol:
            stake = min(stake, available)
            stake_min = float(executor.orch.config.get("risk_management", {}).get("params", {}).get("stake_min", 1.0))
            if stake + 1e-9 < stake_min:
                continue
        if stake <= 0:
            continue

        if i > 0:
            await asyncio.sleep(inter_delay)
            if not signal_epoch_current(executor.orch.config, metrics):
                continue
        recovery_token = None
        unknown_purchase = False
        registered_contract = False
        order_metrics = {**metrics}
        try:
            if setup and metrics.get("_pullback_recovery_amount", 0.0) > 0:
                recovery_token = await claim_recovery(executor.orch, metrics["_pullback_recovery_amount"])
                if recovery_token is None:
                    minimum = float(
                        executor.orch.config.get("risk_management", {}).get("params", {}).get("stake_min", 1.0)
                    )
                    stake, _ = pullback_stake(
                        available, 0.0, float(metrics.get("indicator_quote_rate") or 0.0), minimum
                    )
                    metrics["_pullback_recovery_amount"] = 0.0
                    metrics["indicator_stake_mode"] = "PULLBACK_BASE"
                    if stake <= 0:
                        continue
            custom_dur = metrics.get("duration")
            if custom_dur is None:
                risk_params = executor.orch.config.get("risk_management", {}).get("params", {})
                custom_dur = int(risk_params.get("duration", 300))
            order_metrics = {**metrics, "duration": int(custom_dur), "_register_before_tracking": True}
            if not await claim_indicator_order_once(executor.orch, symbol, order_metrics):
                executor.logger.warning(
                    "EXEC: ordem duplicada evitada | ativo=%s vela=%s",
                    symbol,
                    order_metrics.get("_order_candle_epoch"),
                )
                continue
            if not signal_epoch_current(executor.orch.config, order_metrics):
                await release_unbought_order_claim(executor.orch, symbol, order_metrics)
                continue
            res = await executor._place_order(symbol, direction, stake, duration=custom_dur, metrics=order_metrics)
            if res is None:
                await release_unbought_order_claim(executor.orch, symbol, order_metrics)
            if res:
                registered_contract = True
                if recovery_token is not None:
                    bind_recovery_contract(executor.orch, recovery_token, int(res.contract_id))
                if setup:
                    modes = getattr(executor.orch, "_pullback_contract_modes", None)
                    if not isinstance(modes, dict):
                        modes = {}
                        executor.orch._pullback_contract_modes = modes
                    modes[int(res.contract_id)] = metrics["indicator_stake_mode"]
                executed_stake = float(getattr(res, "buy_price", 0.0) or stake)
                reserved_stake += executed_stake
                registered = await register_confirmed_contract(executor.orch, res, symbol, executed_stake)
                await record_model_attribution(
                    executor.orch, int(res.contract_id), symbol, direction.name, order_metrics, contract=res
                )
                if not registered:
                    executed_count += 1
                    continue
                if order_metrics.get("indicator_tcn_discord"):
                    _discard_opposite_side_training_vectors(executor.orch, str(symbol))
                else:
                    bind_loss_feature_vector_to_contract(executor.orch, str(symbol), int(res.contract_id))
                    bind_loss_flip_ctx_to_contract(executor.orch, str(symbol), int(res.contract_id))
                    meta_vec = order_metrics.get("meta_feature_vector")
                    if isinstance(meta_vec, list) and meta_vec:
                        store_meta_feature_vector(executor.orch, str(symbol), list(meta_vec))
                        bind_meta_feature_vector_to_contract(executor.orch, str(symbol), int(res.contract_id))
                store_contract_audit(
                    executor.orch,
                    int(res.contract_id),
                    symbol=symbol,
                    direction=direction.name,
                    edge=resolve_predicted_edge(order_metrics),
                    meta_payoff_edge_zscore=resolve_meta_payoff_zscore(order_metrics),
                    raw_prob=(
                        float(order_metrics.get("tcn_raw_prob", order_metrics.get("raw_prob")))
                        if order_metrics.get("tcn_raw_prob", order_metrics.get("raw_prob")) is not None
                        else None
                    ),
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
                bm = getattr(executor.orch, "business_metrics", None)
                if bm is not None:
                    try:
                        bm.update_active_contracts_count(len(executor.orch.risk_manager.active_contract_ids))
                        bm.record_indicator_trade(int(res.contract_id), str(symbol), direction.name, order_metrics)
                        if order_metrics.get("direction_origin") != "PULLBACK_SETUP":
                            side_probability = resolve_effective_side_probability(order_metrics, direction.name)
                            if side_probability is not None:
                                bm.track_contract_probability(int(res.contract_id), side_probability)
                            bm.record_inference_radar(
                                str(symbol), direction=direction.name, stake=executed_stake, metrics=order_metrics
                            )
                    except Exception as exc:
                        executor.logger.warning("METRICS: sem telemetria cid=%s: %s", res.contract_id, exc)
        except PurchaseOutcomeUnknownError as e:
            unknown_purchase = True
            await quarantine_unknown_purchase(executor.orch, symbol)
            executor.logger.error("EXEC: compra sem ACK para %s; novas ordens suspensas: %s", symbol, e)
        except Exception as e:
            if handle_broker_maintenance_error(executor.orch, e):
                executor.logger.warning("SKIP: Sessão fechada para %s: %s", symbol, e)
                continue
            err_msg = str(e).lower()
            if "closed" in err_msg or "trading is not available" in err_msg:
                executor.logger.warning("SKIP: Sessão fechada para %s: %s", symbol, e)
            elif is_proposal_runtime_error(e):
                hold = int(
                    executor.orch.config.get("orchestrator", {})
                    .get("execution", {})
                    .get("proposal_failure_skip_cycles", 6)
                )
                executor.orch.risk_manager.register_proposal_failure(symbol, cycles=hold)
        finally:
            if not registered_contract:
                await release_unbought_order_claim(executor.orch, symbol, order_metrics)
            if recovery_token is not None and not unknown_purchase:
                await release_recovery(executor.orch, recovery_token)
    return executed_count

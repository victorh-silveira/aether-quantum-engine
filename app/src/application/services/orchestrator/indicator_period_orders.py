"""Seleciona contratos do indicador por ativo e periodo com cotacao observada."""

from __future__ import annotations

import asyncio
import math
import time

from src.application.services.indicator_consensus import signal_epoch_current
from src.application.services.rise_fall_quote_guard import (
    calculate_payout_breakeven_prob,
    unit_ev,
)
from src.domain.models.trade import TradeDirection
from src.domain.risk.broker_ev_evidence import BrokerEvEvidence
from src.domain.risk.pullback_stake import pullback_stake
from src.infrastructure.handlers.trade_handler_quote_observation import record_indicator_quote
from src.infrastructure.market.timescale_writer import TimescaleMarketWriter


def period_candidates(orch, m5_orders: list[tuple]) -> list[tuple]:
    """Seleciona apenas setups M5 e exclui ativos com contrato aberto."""
    active_ids = set(getattr(orch.state, "active_contracts", {}))
    active_symbols = {
        symbol
        for contract_id, symbol in getattr(orch.risk_manager, "contract_to_symbol", {}).items()
        if contract_id in active_ids
    }
    result = [item for item in m5_orders if item[0] not in active_symbols]
    mode = str(orch.config.get("trading", {}).get("mode", "demo")).lower()
    evidence_by_version = getattr(orch, "_indicator_broker_evidence", {})
    real_by_version = getattr(orch, "_indicator_real_evidence", {})
    writer = getattr(orch, "market_writer", None)
    if not isinstance(writer, TimescaleMarketWriter) or writer._last_error is not None:
        return []
    excluded = set(
        orch.config.get("pullback_strategy", {}).get("excluded_symbols")
        or orch.config.get("strategy", {}).get("excluded_symbols")
        or []
    )
    for (symbol, period), signal in getattr(orch, "_indicator_period_signals", {}).items():
        if period != 300 or symbol in active_symbols or symbol in excluded:
            continue
        if not signal.model_version.startswith("pullback-1-"):
            continue
        evidence = evidence_by_version.get((symbol, period, signal.model_version), BrokerEvEvidence(0, ()))
        pilot = mode == "demo" and evidence.pilot_available and not evidence.promoted
        if not pilot and (mode != "real" or not evidence.promoted):
            continue
        if (
            mode == "real"
            and real_by_version.get((symbol, period, signal.model_version), BrokerEvEvidence(0, ())).suspended
        ):
            continue
        win_count = sum(value > 0 for value in evidence.returns)
        prob_estimate = (win_count + 0.5) / (evidence.settled + 1.0) if evidence.settled > 0 else 0.5
        probability = evidence.posterior_win_probability or prob_estimate if mode == "real" else prob_estimate
        call_probability = probability if signal.side == "CALL" else 1.0 - probability
        metrics = {
            "indicator_controls_execution": True,
            "indicator_candle_epoch": signal.candle_epoch,
            "indicator_period_seconds": period,
            "indicator_model_version": signal.model_version,
            "indicator_call_probability": None,
            "indicator_execution_side": signal.side,
            "indicator_demo_pilot": pilot,
            "indicator_broker_settled": evidence.settled,
            "indicator_broker_lower_95": evidence.lower_95,
            "indicator_exploratory_purchased": evidence.exploratory_purchased,
            "calibrated_prob": call_probability,
            "raw_prob": None,
            "trade_score": probability,
            "conviction": probability,
            "senior_trader_conviction": probability,
            "direction_origin": "PULLBACK_SETUP",
            "indicator_tcn_discord": True,
            "duration": 5,
        }
        if signal_epoch_current(orch.config, metrics):
            result.append((symbol, TradeDirection[signal.side], metrics))
    return result


async def _prequote(orch, order: tuple, semaphore: asyncio.Semaphore) -> tuple | None:
    """Mede EV cotado e registra a observacao antes da escolha."""
    symbol, direction, metrics = order
    if not signal_epoch_current(orch.config, metrics):
        return None
    period = int(metrics.get("indicator_period_seconds") or 300)
    if period != 300:
        return None
    duration = period // 60
    stake_min = float(orch.config.get("risk_management", {}).get("params", {}).get("stake_min", 1.0))
    stake, _ = pullback_stake(float(orch.state.balance), 0.0, 1.0, stake_min)
    if stake <= 0:
        return None
    params = {"duration": duration, "duration_unit": "m"}
    async with semaphore:
        quote = await asyncio.wait_for(
            orch.trade_handler.fetch_proposal_snapshot(symbol, direction, stake, params=params), timeout=4.0
        )
    if quote is None or not quote.get("proposal_id") or not signal_epoch_current(orch.config, metrics):
        return None
    rate = float(quote["rate"])
    payout_break_even = calculate_payout_breakeven_prob(rate)
    metrics["indicator_quote_rate"] = rate
    pilot = bool(metrics.get("indicator_demo_pilot"))
    if pilot and payout_break_even is not None:
        await record_indicator_quote(
            orch.market_writer,
            orch.config,
            symbol,
            direction.name,
            {
                "_indicator_model_version": metrics["indicator_model_version"],
                "_indicator_period_seconds": period,
                "_signal_candle_epoch": metrics["indicator_candle_epoch"],
            },
            str(quote["proposal_id"]),
            float(quote["price"]),
            float(quote["payout"]),
            rate,
        )
        metrics["indicator_prequote_ev"] = None
        return order
    haircut = float(orch.config.get("orchestrator", {}).get("execution", {}).get("quote_probability_haircut", 0.0))
    probability = float(metrics["trade_score"]) - haircut
    edge = unit_ev(probability, rate)
    if edge is None or payout_break_even is None:
        return None
    confirmatory = edge > 0 and probability >= payout_break_even + 0.01
    await record_indicator_quote(
        orch.market_writer,
        orch.config,
        symbol,
        direction.name,
        {
            "_indicator_model_version": metrics["indicator_model_version"],
            "_indicator_period_seconds": period,
            "_signal_candle_epoch": metrics["indicator_candle_epoch"],
            "_quote_guard_side_probability": probability,
        },
        str(quote["proposal_id"]),
        float(quote["price"]),
        float(quote["payout"]),
        rate,
    )
    mode = str(orch.config.get("trading", {}).get("mode", "demo")).lower()
    if mode == "real" and not confirmatory:
        return None
    if mode == "demo" and not pilot:
        return None
    metrics["indicator_prequote_ev"] = edge
    return order


async def select_period_orders(orch, m5_orders: list[tuple]) -> list[tuple]:
    """Escolhe um periodo por ativo pelo limite DEMO e depois EV cotado."""
    candidates = period_candidates(orch, m5_orders)
    if not candidates:
        return []
    semaphore = asyncio.Semaphore(8)
    tasks = [asyncio.create_task(_prequote(orch, order, semaphore)) for order in candidates]
    deadline = max(0.0, min(20.0, 30.0 - time.time() % 300))
    done, pending = await asyncio.wait(tasks, timeout=deadline)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    quoted = [task.result() for task in done if not task.cancelled() and task.exception() is None and task.result()]
    chosen = {}
    for order in quoted:
        symbol, _, metrics = order
        score = (
            float(metrics["indicator_broker_lower_95"])
            if metrics.get("indicator_broker_lower_95") is not None
            else -math.inf,
            float(metrics.get("trade_score") or metrics.get("calibrated_prob") or 0.5),
            float(metrics["indicator_prequote_ev"])
            if metrics.get("indicator_prequote_ev") is not None
            else float(metrics.get("indicator_quote_rate") or 0.0),
            float(metrics.get("indicator_quote_rate") or 0.0),
        )
        if symbol not in chosen or score > chosen[symbol][0]:
            chosen[symbol] = (score, order)
    multi_enabled = bool(orch.config.get("orchestrator", {}).get("execution", {}).get("multi_symbol_orders", False))
    maximum = (
        int(orch.config.get("orchestrator", {}).get("execution", {}).get("max_orders_per_cycle", 1))
        if multi_enabled
        else 1
    )
    ranked = [item[1] for item in sorted(chosen.values(), key=lambda item: item[0], reverse=True)]
    return ranked[: max(1, maximum)]

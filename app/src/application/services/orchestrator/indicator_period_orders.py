"""Seleciona contratos do indicador por ativo e periodo com cotacao observada."""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

from src.application.services.indicator_consensus import signal_epoch_current
from src.application.services.orchestrator.indicator_period_filters import (
    build_indicator_metrics,
    closed_candles_slice,
    is_promoted_signal_viable,
)
from src.application.services.rise_fall_quote_guard import calculate_payout_breakeven_prob, unit_ev
from src.domain.models.trade import TradeDirection
from src.domain.risk.broker_ev_evidence import BrokerEvEvidence
from src.domain.risk.pullback_stake import pullback_stake
from src.infrastructure.handlers.trade_handler_quote_observation import record_indicator_quote
from src.infrastructure.market.timescale_writer import TimescaleMarketWriter


def _closed_candles_slice(candles: list[Any], period: int, now_ts: float) -> list[Any]:
    """Compatibilidade interna para fatiamento de velas fechadas."""
    return closed_candles_slice(candles, period, now_ts)


def period_candidates(orch: Any, m5_orders: list[tuple]) -> list[tuple]:
    """Seleciona apenas setups do indicador e exclui ativos com contrato aberto."""
    active_ids = set(getattr(orch.state, "active_contracts", {}))
    active_symbols = {
        symbol
        for contract_id, symbol in getattr(orch.risk_manager, "contract_to_symbol", {}).items()
        if contract_id in active_ids
    }
    pb_cfg = orch.config.get("pullback_strategy", {})
    pb_enabled = bool(pb_cfg.get("enabled", False))
    result = [] if pb_enabled else [item for item in m5_orders if item[0] not in active_symbols]
    mode = str(orch.config.get("trading", {}).get("mode", "demo")).lower()
    evidence_by_version = getattr(orch, "_indicator_broker_evidence", {})
    real_by_version = getattr(orch, "_indicator_real_evidence", {})
    writer = getattr(orch, "market_writer", None)
    if isinstance(writer, TimescaleMarketWriter) and writer._last_error is not None:
        return []
    excluded = set(pb_cfg.get("excluded_symbols") or orch.config.get("strategy", {}).get("excluded_symbols") or [])
    promoted_only = pb_cfg.get("promoted_symbol") or orch.config.get("anchor")
    dynamic_rotation, period_seconds = (
        bool(pb_cfg.get("dynamic_rotation", False)),
        int(pb_cfg.get("period_seconds", 300)),
    )
    now_ts = time.time()
    has_promoted_signal = (
        is_promoted_signal_viable(orch, promoted_only, period_seconds, now_ts, signal_epoch_fn=signal_epoch_current)
        if (pb_enabled and dynamic_rotation)
        else False
    )
    quarantine = getattr(orch, "_symbol_loss_quarantine", {})
    operate_all = bool(pb_cfg.get("operate_all_signals", False))
    for (symbol, period), signal in getattr(orch, "_indicator_period_signals", {}).items():
        if (
            pb_enabled
            and promoted_only
            and not operate_all
            and symbol != str(promoted_only)
            and (not dynamic_rotation or has_promoted_signal)
        ):
            continue
        if (
            period != period_seconds
            or symbol in active_symbols
            or symbol in excluded
            or now_ts < quarantine.get(symbol, 0)
            or (
                signal.model_version != pb_cfg.get("version")
                if pb_enabled
                else not signal.model_version.startswith("pullback-1-")
            )
        ):
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
        side = signal.side
        decision_reason = "sinal_base"
        if side not in ("CALL", "PUT"):
            continue
        metrics = build_indicator_metrics(
            signal.candle_epoch,
            period,
            signal.model_version,
            side,
            signal.side,
            decision_reason,
            pilot=pilot,
            ev=evidence,
            prob=probability,
        )
        if signal_epoch_current(orch.config, metrics):
            result.append((symbol, TradeDirection[side], metrics))
    return result


async def _record_quote(
    orch: Any, symbol: str, name: str, metrics: dict, period: int, quote: dict, rate: float, prob: float | None = None
) -> None:
    """Grava observacao de cotacao no Timescale."""
    payload = {
        "_indicator_model_version": metrics.get("indicator_model_version", ""),
        "_indicator_period_seconds": period,
        "_signal_candle_epoch": metrics.get("indicator_candle_epoch", 0),
    }
    if prob is not None:
        payload["_quote_guard_side_probability"] = prob
    p_id, pr, po = str(quote["proposal_id"]), float(quote["price"]), float(quote["payout"])
    await record_indicator_quote(orch.market_writer, orch.config, symbol, name, payload, p_id, pr, po, rate)


async def _prequote(orch: Any, order: tuple, semaphore: asyncio.Semaphore) -> tuple | None:
    """Mede EV cotado e registra a observacao antes da escolha."""
    symbol, direction, metrics = order
    if not signal_epoch_current(orch.config, metrics):
        return None
    period = int(metrics.get("indicator_period_seconds") or 300)
    allowed_period = int(orch.config.get("pullback_strategy", {}).get("period_seconds", 300))
    if period != allowed_period:
        return None
    stake_min = float(orch.config.get("risk_management", {}).get("params", {}).get("stake_min", 1.0))
    stake, _ = pullback_stake(float(orch.state.balance), 0.0, 1.0, stake_min)
    if stake <= 0:
        return None
    params = {"duration": period // 60, "duration_unit": "m"}
    async with semaphore:
        quote = await asyncio.wait_for(
            orch.trade_handler.fetch_proposal_snapshot(symbol, direction, stake, params=params), timeout=4.0
        )
    if quote is None or not quote.get("proposal_id") or not signal_epoch_current(orch.config, metrics):
        return None
    rate = float(quote["rate"])
    default_min_rate = 0.68 if str(symbol).startswith("OTC_") else 0.75
    min_rate = float(orch.config.get("pullback_strategy", {}).get("min_payout_rate", default_min_rate))
    if rate < min_rate:
        return None
    payout_break_even = calculate_payout_breakeven_prob(rate)
    metrics.update(
        {
            "indicator_quote_rate": rate,
            "_prequoted_proposal_id": str(quote["proposal_id"]),
            "_prequoted_ask_price": float(quote["price"]),
            "_prequoted_payout": float(quote["payout"]),
            "_prequoted_rate": rate,
            "_prequoted_timestamp": time.time(),
        }
    )
    pilot = bool(metrics.get("indicator_demo_pilot"))
    if pilot and payout_break_even is not None:
        await _record_quote(orch, symbol, direction.name, metrics, period, quote, rate)
        metrics["indicator_prequote_ev"] = None
        return order
    haircut = float(orch.config.get("orchestrator", {}).get("execution", {}).get("quote_probability_haircut", 0.0))
    probability = float(metrics["trade_score"]) - haircut
    edge = unit_ev(probability, rate)
    if edge is None or payout_break_even is None:
        return None
    confirmatory = edge > 0 and probability >= payout_break_even + 0.01
    await _record_quote(orch, symbol, direction.name, metrics, period, quote, rate, prob=probability)
    mode = str(orch.config.get("trading", {}).get("mode", "demo")).lower()
    if mode == "demo" and not pilot:
        return None
    if mode == "real" and not confirmatory:
        return None
    metrics["indicator_prequote_ev"] = edge
    return order


async def select_period_orders(orch: Any, m5_orders: list[tuple]) -> list[tuple]:
    """Escolhe um periodo por ativo pelo limite DEMO e depois EV cotado."""
    candidates = period_candidates(orch, m5_orders)
    if not candidates:
        return []
    semaphore = asyncio.Semaphore(8)
    tasks = [asyncio.create_task(_prequote(orch, order, semaphore)) for order in candidates]
    period_seconds = int(orch.config.get("pullback_strategy", {}).get("period_seconds", 300))
    window = float(orch.config.get("pullback_strategy", {}).get("entry_window_seconds", 60.0))
    deadline = max(2.0, min(20.0, window - time.time() % period_seconds))
    done, pending = await asyncio.wait(tasks, timeout=deadline)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    quoted = [task.result() for task in done if not task.cancelled() and task.exception() is None and task.result()]
    chosen: dict[str, tuple[tuple, tuple]] = {}
    for order in quoted:
        symbol, _, metrics = order
        l95, ev_val = metrics.get("indicator_broker_lower_95"), metrics.get("indicator_prequote_ev")
        rt = float(metrics.get("indicator_quote_rate") or 0.0)
        t_str = float(metrics.get("trend_strength") or 0.0)
        pen = -0.05 if getattr(orch, "_last_traded_symbol", None) == symbol else 0.0
        score = (
            float(l95) if l95 is not None else -math.inf,
            float(metrics.get("trade_score") or metrics.get("calibrated_prob") or 0.5) + pen,
            float(ev_val) if ev_val is not None else rt,
            t_str,
            rt,
        )
        if symbol not in chosen or score > chosen[symbol][0]:
            chosen[symbol] = (score, order)
    multi = bool(orch.config.get("orchestrator", {}).get("execution", {}).get("multi_symbol_orders", False))
    lim = int(orch.config.get("orchestrator", {}).get("execution", {}).get("max_orders_per_cycle", 1)) if multi else 1
    ranked = [item[1] for item in sorted(chosen.values(), key=lambda item: item[0], reverse=True)]
    if not multi and orch.config.get("pullback_strategy", {}).get("enabled"):
        fb_depth = int(orch.config.get("pullback_strategy", {}).get("max_fallback_candidates", 1))
        return ranked[: max(1, fb_depth)]
    return ranked[: max(1, lim)]

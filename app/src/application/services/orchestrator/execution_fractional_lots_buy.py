"""Compra de sub-lotes fracionados a partir de proposta Deriv."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.application.services.market_audit_log import (
    format_execution_ticket_line,
    resolve_meta_payoff_zscore,
    resolve_predicted_edge,
    resolve_stake_audit_context,
    store_contract_audit,
)
from src.application.services.orchestrator.settlement_backfill import subscribe_open_contract
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.infrastructure.handlers.trade_handler import _contract_duration_seconds, build_proposal_request


async def buy_lot_from_proposal(
    executor: Any,
    symbol: str,
    direction: TradeDirection,
    lot: float,
    proposal: dict[str, Any],
    *,
    duration: int | None,
    metrics: dict[str, Any],
) -> Contract:
    """Compra um sub-lote com token de proposta previamente reservado."""
    timeout = int(executor.orch.ws.request_timeout)
    proposal_id = str(proposal["id"])
    ask_price = float(proposal.get("ask_price") or lot)
    buy_response = await executor.orch.ws.send({"buy": proposal_id, "price": ask_price}, timeout=timeout)
    if not isinstance(buy_response, dict):
        raise RuntimeError("Erro na compra direta: resposta invalida")
    if "error" in buy_response:
        raise RuntimeError(f"Erro na compra direta: {buy_response['error'].get('message', 'Erro desconhecido')}")
    buy = buy_response.get("buy")
    if not isinstance(buy, dict):
        raise RuntimeError("Erro na compra direta: resposta sem buy")
    params = executor.orch.config.get("risk_management", {}).get("params", {}).copy()
    if duration is not None:
        params["duration"] = int(duration)
    expiry = int(proposal.get("date_expiry") or buy.get("date_expiry") or 0)
    if expiry <= 0:
        expiry = int(time.time()) + _contract_duration_seconds(build_proposal_request(symbol, direction, lot, params))
    contract = Contract(
        contract_id=int(buy["contract_id"]),
        proposal_id=proposal_id,
        status=TradeStatus.OPEN,
        buy_price=float(buy.get("buy_price") or ask_price),
        payout=float(buy.get("payout") or proposal.get("payout") or 0.0),
        symbol=symbol,
        direction=direction,
        stake=float(lot),
        expiry_time=expiry,
        longcode=str(buy.get("longcode") or proposal.get("longcode") or ""),
    )
    cycle_id = int(getattr(executor.orch, "_active_cycle_id", 0))
    audit = resolve_stake_audit_context(executor.orch.risk_manager)
    executor.logger.info(
        format_execution_ticket_line(
            cycle_id,
            direction=direction.name,
            symbol=str(symbol),
            stake=float(lot),
            mode_tag=str(audit.get("mode_tag") or "EXPLORE_KELLY"),
            pending=float(audit.get("pending", 0.0)),
            bankroll=float(audit.get("bankroll", 0.0)),
            contract_id=int(contract.contract_id),
            payout=float(contract.payout),
            linear=int(audit.get("linear", 0)),
            cap=float(audit.get("cap", 0.0)),
            recovery_infeasible=bool(audit.get("recovery_infeasible", False)),
        )
    )
    store_contract_audit(
        executor.orch,
        int(contract.contract_id),
        symbol=str(symbol),
        direction=direction.name,
        edge=resolve_predicted_edge(metrics if isinstance(metrics, dict) else {}),
        meta_payoff_edge_zscore=resolve_meta_payoff_zscore(metrics if isinstance(metrics, dict) else None),
        raw_prob=(
            float(metrics["raw_prob"]) if isinstance(metrics, dict) and metrics.get("raw_prob") is not None else None
        ),
    )
    executor.orch.risk_manager.contract_to_symbol[contract.contract_id] = symbol
    req_timeout = float(
        executor.orch.config.get("orchestrator", {})
        .get("execution", {})
        .get("settlement_request_timeout_seconds", 30.0)
    )
    asyncio.create_task(subscribe_open_contract(executor.orch.ws, int(contract.contract_id), timeout=req_timeout))
    return contract

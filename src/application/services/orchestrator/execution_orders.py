"""Envio continuo de ordens Rise/Fall 5m e inscricao de liquidacao."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.application.services.orchestrator.api_maintenance_guard import handle_broker_maintenance_error
from src.application.services.orchestrator.execution_proposal import (
    is_retriable_proposal_error,
    proposal_retry_scales,
    proposal_stake_attempts,
)
from src.application.services.orchestrator.settlement_backfill import subscribe_open_contract
from src.application.services.orchestrator.settlement_rest import schedule_rest_contract_settlement
from src.domain.models.trade import Contract, TradeDirection
from src.infrastructure.telemetry.otel_tracer import trace_span


async def _subscribe_open_contract_background(ws: Any, contract_id: int, *, timeout: float, cid: str) -> None:
    """Inscreve liquidacao em background para nao bloquear o ciclo."""
    logger = logging.getLogger("AETH")
    try:
        await subscribe_open_contract(ws, int(contract_id), timeout=timeout)
    except Exception as exc:
        logger.warning("[%s] SETTLE.settle_subscribe: subscribe cid=%s falhou: %s", cid, int(contract_id), exc)


async def place_order(
    executor: Any,
    symbol: str,
    direction: TradeDirection,
    stake: float,
    duration: int = 5,
    metrics: dict[str, Any] | None = None,
) -> Contract | None:
    """Envia proposta e compra Rise/Fall 5m via Deriv e agenda liquidacao."""
    logger = logging.getLogger("AETH")
    cid = f"C{int(getattr(executor.orch, '_active_cycle_id', 1)):04d}"
    exec_cfg = executor.orch.config.get("orchestrator", {}).get("execution", {})
    risk_cfg = executor.orch.config.get("risk_management", {})
    stake_min = float(risk_cfg.get("params", {}).get("stake_min", 1.0))
    params: dict[str, Any] = {
        "contract_type": "CALL",
        "duration": int(duration),
        "duration_unit": "m",
        "stake_min": stake_min,
    }
    if metrics and "laya_probability" in metrics:
        params["_laya_sizing"] = {
            "probability": metrics["laya_probability"],
            "pending": executor.orch.risk_manager.pending_loss,
            "config": risk_cfg,
            "on_balance": lambda bal: setattr(executor.orch.state, "balance", float(bal)),
        }
        params["_indicator_model_version"] = metrics.get("laya_model_version", "convaiinnovations/laya")
    attempts = proposal_stake_attempts(float(stake), stake_min, proposal_retry_scales(exec_cfg))
    if not attempts:
        return None
    contract: Contract | None = None
    last_error: Exception | None = None
    with trace_span("trade.place_order", attributes={"symbol": str(symbol), "direction": direction.name}):
        for attempt_stake in attempts:
            try:
                contract = await executor.orch.trade_handler.buy_with_parameters(
                    symbol, direction, attempt_stake, params=params
                )
                stake = float(attempt_stake)
                break
            except RuntimeError as exc:
                last_error = exc
                if handle_broker_maintenance_error(executor.orch, exc):
                    raise
                if not is_retriable_proposal_error(exc) or attempt_stake == attempts[-1]:
                    raise
    if contract is None:
        raise last_error or RuntimeError("Erro na proposta: falha desconhecida")
    logger.info(
        "EXEC | %s %s | stake=$%.2f | payout=$%.2f | cid=%s",
        symbol,
        direction.name,
        float(contract.buy_price or stake),
        float(contract.payout or 0.0),
        int(contract.contract_id),
    )
    inst = getattr(executor.orch, "business_instrumentor", None)
    if inst is not None:
        inst.record_cycle(str(symbol), is_execution=True, reason="laya_continuo")
        if hasattr(inst, "runtime"):
            inst.runtime.record_quote_payout(str(symbol), float(contract.buy_price), float(contract.payout))
    executor.orch.risk_manager.contract_to_symbol[contract.contract_id] = symbol
    req_timeout = float(exec_cfg.get("settlement_request_timeout_seconds", 30.0))
    if str(getattr(executor.orch, "trading_transport", "ws")).lower() == "rest":
        base_bal = float(getattr(executor.orch.state, "balance", 0.0) or 0.0)
        balance_after = base_bal - float(contract.buy_price)
        try:
            client = executor.orch.auth.rest_client()
            accounts = await client.list_accounts()
            account_id = str(getattr(executor.orch, "deriv_account_id", "") or "")
            for acc in accounts:
                if not account_id or str(acc.account_id) == account_id:
                    fetched = float(acc.balance)
                    balance_after = fetched if fetched < base_bal else balance_after
                    executor.orch.state.balance = balance_after
                    break
        except Exception as exc:
            logger.warning("[%s] SETTLE_REST: refresh saldo pos-compra falhou: %s", cid, exc)
        schedule_rest_contract_settlement(executor.orch, contract, balance_after_buy=balance_after)
    else:
        asyncio.create_task(
            _subscribe_open_contract_background(
                executor.orch.ws, int(contract.contract_id), timeout=req_timeout, cid=cid
            )
        )
    return contract

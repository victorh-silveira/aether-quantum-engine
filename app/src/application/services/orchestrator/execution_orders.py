"""Envio de ordens e inscricao em atualizacoes de contrato aberto."""

import asyncio
import logging

from src.application.services.market_audit_log import format_direction_audit_line, resolve_predicted_edge
from src.domain.risk.stop_win_target import resolve_stop_win_target

from .execution_proposal import (
    is_retriable_proposal_error,
    proposal_retry_scales,
    proposal_stake_attempts,
)
from .settlement_backfill import subscribe_open_contract


async def _subscribe_open_contract_background(ws, contract_id: int, *, timeout: float, cid: str) -> None:
    """Inscreve liquidacao em background para nao bloquear o ciclo de execucao."""
    logger = logging.getLogger("AETH")
    try:
        await subscribe_open_contract(ws, int(contract_id), timeout=timeout)
    except Exception as e:
        logger.warning("[%s] SETTLE: subscribe cid=%s falhou: %s", cid, int(contract_id), e)


async def place_order(executor, symbol, direction, stake, duration=None, metrics=None):
    """Compra contrato com parametros de risco e registra assinatura de liquidacao."""
    cid = f"C{int(executor.orch._active_cycle_id):04d}"
    logger = logging.getLogger("AETH")
    params = executor.orch.config.get("risk_management", {}).get("params", {}).copy()
    if duration:
        params["duration"] = duration
    if params.get("contract_type") == "MULTIPLIER":
        target_total = resolve_stop_win_target(
            executor.orch.config.get("risk_management"), executor.orch.risk_manager.initial_bankroll
        )
        current_profit = executor.orch.risk_manager.total_session_profit
        remaining = target_total - current_profit
        if remaining > 0:
            lo = dict(params.get("limit_order") or {})
            max_tp = float(stake) * 50.0
            tp_val = min(float(remaining), max_tp)
            lo["take_profit"] = round(tp_val, 2)
            lo.pop("stop_loss", None)
            params["limit_order"] = lo
    exec_cfg = executor.orch.config.get("orchestrator", {}).get("execution", {})
    stake_min = float(params.get("stake_min", 1.0))
    attempts = proposal_stake_attempts(float(stake), stake_min, proposal_retry_scales(exec_cfg))
    contract = None
    last_error: Exception | None = None
    for attempt_stake in attempts:
        try:
            contract = await executor.orch.trade_handler.buy_with_parameters(
                symbol, direction, attempt_stake, params=params
            )
            if attempt_stake + 1e-9 < float(stake):
                logger.info(
                    "[%s] PROPOSAL_RETRY | %s stake $%.2f -> $%.2f",
                    cid,
                    symbol,
                    float(stake),
                    float(attempt_stake),
                )
            stake = float(attempt_stake)
            break
        except RuntimeError as exc:
            last_error = exc
            if not is_retriable_proposal_error(exc) or attempt_stake == attempts[-1]:
                raise
    if contract is None:
        raise last_error or RuntimeError("Erro na proposta: falha desconhecida")
    dur = duration or params.get("duration", 1)
    u = params.get("duration_unit", "m")
    meta = metrics if isinstance(metrics, dict) else {}
    dl_dir = meta.get("dl_direction")
    cycle_id = int(executor.orch._active_cycle_id)
    logger.info(
        "%s || stake=$%.2f pay=%.2f cid=%s buy=$%.2f %s%s",
        format_direction_audit_line(
            cycle_id,
            direction.name,
            str(symbol),
            resolve_predicted_edge(meta),
            dl_direction=str(dl_dir) if dl_dir else None,
        ),
        float(stake),
        float(contract.payout),
        int(contract.contract_id),
        float(contract.buy_price),
        str(dur),
        str(u),
    )
    executor.orch.risk_manager.contract_to_symbol[contract.contract_id] = symbol
    req_timeout = float(
        executor.orch.config.get("orchestrator", {})
        .get("execution", {})
        .get("settlement_request_timeout_seconds", 30.0)
    )
    asyncio.create_task(
        _subscribe_open_contract_background(
            executor.orch.ws,
            int(contract.contract_id),
            timeout=req_timeout,
            cid=cid,
        )
    )
    return contract

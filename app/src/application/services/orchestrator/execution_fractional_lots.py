"""Fatiamento de stakes elevadas em lotes paralelos para a API Deriv."""

from __future__ import annotations

import asyncio
import math
import secrets
import uuid
from typing import Any

from src.application.services.orchestrator.execution_contract_adoption import adopt_executed_contract
from src.application.services.orchestrator.execution_fractional_lots_buy import buy_lot_from_proposal
from src.application.services.orchestrator.execution_split_abort import next_split_attempt_seq
from src.domain.models.trade import Contract, TradeDirection
from src.infrastructure.handlers.trade_handler import build_proposal_request


MAX_SINGLE_STAKE_LIMIT = 200.0
FRACTIONAL_STAGGER_BASE_US = 250.0
FRACTIONAL_STAGGER_JITTER_US = 150.0
FRACTIONAL_STAGGER_RTT_REFERENCE_SECONDS = 0.08

_buy_lot_from_proposal = buy_lot_from_proposal


def resolve_fractional_lot_stagger_seconds(orch: Any, exec_cfg: dict[str, Any] | None = None) -> float:
    """Calcula atraso estocastico entre sub-propostas com escala pelo RTT do WebSocket."""
    chunk = exec_cfg if isinstance(exec_cfg, dict) else {}
    if not chunk:
        orch_cfg = getattr(orch, "config", {})
        chunk = orch_cfg.get("orchestrator", {}).get("execution", {}) if isinstance(orch_cfg, dict) else {}
    base_us = float(chunk.get("fractional_lot_stagger_base_us", FRACTIONAL_STAGGER_BASE_US))
    jitter_us = float(chunk.get("fractional_lot_stagger_jitter_us", FRACTIONAL_STAGGER_JITTER_US))
    ws = getattr(orch, "ws", None)
    rtt = float(getattr(ws, "last_rtt_seconds", 0.0) or 0.0)
    rtt_scale = max(0.5, min(2.0, rtt / FRACTIONAL_STAGGER_RTT_REFERENCE_SECONDS)) if rtt > 0.0 else 1.0
    jitter_span = max(0.0, jitter_us)
    jitter = secrets.randbelow(10_000) / 10_000.0 * jitter_span if jitter_span > 0.0 else 0.0
    return max(0.0, ((base_us + jitter) * rtt_scale) / 1_000_000.0)


async def _stagger_fractional_dispatch(orch: Any, lot_index: int, exec_cfg: dict[str, Any] | None = None) -> None:
    """Aplica jitter controlado entre sub-lotes para reduzir colisao no gateway WS."""
    if lot_index > 0:
        delay = resolve_fractional_lot_stagger_seconds(orch, exec_cfg)
        if delay > 0.0:
            await asyncio.sleep(delay)


def resolve_max_single_stake_limit(exec_cfg: dict[str, Any] | None) -> float:
    """Retorna teto configuravel de stake por boleta unica."""
    return max(0.01, float((exec_cfg or {}).get("max_single_stake_limit", MAX_SINGLE_STAKE_LIMIT)))


def split_fractional_stake_lots(stake: float, *, limit: float) -> list[float]:
    """Divide stake total em N lotes <= limit com soma exata em centavos."""
    total = round(float(stake), 2)
    if total <= float(limit) + 1e-9:
        return [total]
    lot_count = int(math.ceil(total / float(limit)))
    base = round(total / lot_count, 2)
    lots = [base] * (lot_count - 1)
    lots.append(round(total - sum(lots), 2))
    return [lot for lot in lots if lot > 0.0]


def register_contract_lot_group(orch: Any, contract_ids: list[int]) -> int:
    """Vincula contratos fracionados ao ciclo corrente para liquidacao agrupada."""
    group_id = int(getattr(orch, "_active_cycle_id", 0))
    normalized = tuple(int(cid) for cid in contract_ids)
    if not isinstance(getattr(orch, "_contract_lot_groups", None), dict):
        orch._contract_lot_groups = {}
    orch._contract_lot_groups[group_id] = normalized
    if not isinstance(getattr(orch, "_contract_lot_group", None), dict):
        orch._contract_lot_group = {}
    for cid in normalized:
        orch._contract_lot_group[int(cid)] = group_id
    return group_id


async def dispatch_fractional_orders(
    executor: Any,
    symbol: str,
    direction: TradeDirection,
    stake: float,
    *,
    duration: int | None,
    metrics: dict[str, Any],
    order_n: int,
) -> list[Any]:
    """Dispara uma ou N ordens paralelas respeitando o teto de stake por boleta."""
    exec_cfg = executor.orch.config.get("orchestrator", {}).get("execution", {})
    limit = resolve_max_single_stake_limit(exec_cfg if isinstance(exec_cfg, dict) else {})
    lots = split_fractional_stake_lots(stake, limit=limit)
    place = executor._place_order
    if len(lots) == 1:
        contract = await place(symbol, direction, lots[0], duration=duration, metrics=metrics)
        if not contract:
            return []
        await adopt_executed_contract(
            executor,
            contract,
            symbol=symbol,
            direction=direction,
            metrics=metrics,
            requested_stake=lots[0],
            order_n=order_n,
        )
        return [contract]
    contracts = await _dispatch_split_lot_orders_atomically(
        executor, symbol, direction, lots, duration=duration, metrics=metrics
    )
    if not contracts:
        return []
    register_contract_lot_group(executor.orch, [int(c.contract_id) for c in contracts])
    for lot_stake, contract in zip(lots, contracts, strict=False):
        await adopt_executed_contract(
            executor,
            contract,
            symbol=symbol,
            direction=direction,
            metrics=metrics,
            requested_stake=float(lot_stake),
            order_n=order_n,
        )
    return contracts


async def _dispatch_split_lot_orders_atomically(
    executor: Any,
    symbol: str,
    direction: TradeDirection,
    lots: list[float],
    *,
    duration: int | None,
    metrics: dict[str, Any],
) -> list[Contract]:
    """Executa split em duas fases: proposta completa e compra subsequente."""
    proposal_batch = await _prepare_split_lot_proposals(executor, symbol, direction, lots, duration=duration)
    if proposal_batch is None:
        metrics["fractional_lot_technical_failure"] = True
        return []
    contracts: list[Contract] = []
    for lot, proposal in zip(lots, proposal_batch, strict=True):
        contract = await _buy_lot_from_proposal(
            executor, symbol, direction, lot, proposal, duration=duration, metrics=metrics
        )
        contracts.append(contract)
    return contracts


async def _prepare_split_lot_proposals(
    executor: Any,
    symbol: str,
    direction: TradeDirection,
    lots: list[float],
    *,
    duration: int | None,
) -> list[dict[str, Any]] | None:
    """Solicita proposta unica por sub-lote e aborta em qualquer falha."""
    params = executor.orch.config.get("risk_management", {}).get("params", {}).copy()
    if duration is not None:
        params["duration"] = int(duration)
    timeout = int(executor.orch.ws.request_timeout)
    proposal_rows: list[dict[str, Any]] = []
    proposal_ids: set[str] = set()
    split_batch_id = uuid.uuid4().hex
    split_attempt_seq = next_split_attempt_seq(executor.orch)
    cycle_id = int(getattr(executor.orch, "_active_cycle_id", 0))
    executor.logger.info(
        "[C%04d] EXEC_SPLIT_START | %s %s | lots=%d | seq=%d | batch=%s",
        cycle_id,
        symbol,
        direction.name,
        len(lots),
        split_attempt_seq,
        split_batch_id,
    )
    split_lot_id = ""
    exec_cfg = executor.orch.config.get("orchestrator", {}).get("execution", {})
    for index, lot in enumerate(lots):
        if index > 0:
            await _stagger_fractional_dispatch(executor.orch, index, exec_cfg if isinstance(exec_cfg, dict) else {})
        try:
            proposal_req = build_proposal_request(symbol, direction, lot, params)
            passthrough = proposal_req.get("passthrough")
            passthrough_dict = dict(passthrough) if isinstance(passthrough, dict) else {}
            split_lot_id = f"{split_batch_id}_split_{index}"
            passthrough_dict["split_batch_id"] = split_batch_id
            passthrough_dict["split_attempt_seq"] = split_attempt_seq
            passthrough_dict["split_lot_index"] = int(index)
            passthrough_dict["split_lot_id"] = split_lot_id
            proposal_req["passthrough"] = passthrough_dict
            response = await executor.orch.ws.send(proposal_req, timeout=timeout)
            if not isinstance(response, dict):
                raise RuntimeError("Erro na proposta: resposta invalida")
            proposal = response.get("proposal") if isinstance(response, dict) else None
            if "error" in response:
                raise RuntimeError(f"Erro na proposta: {response['error'].get('message', 'Erro desconhecido')}")
            if not isinstance(proposal, dict):
                raise RuntimeError("Erro na proposta: resposta sem proposal")
            proposal_id = str(proposal.get("id") or "")
            if not proposal_id:
                raise RuntimeError("Erro na proposta: id ausente")
            if proposal_id in proposal_ids:
                raise RuntimeError("Erro na proposta: id duplicado em fatiamento")
            proposal_ids.add(proposal_id)
            proposal_rows.append(proposal)
        except Exception as exc:
            executor.logger.warning(
                "[C%04d] EXEC_SPLIT_ABORT | %s %s lote=$%.2f | seq=%d | lot=%s | batch=%s | %s",
                cycle_id,
                symbol,
                direction.name,
                float(lot),
                split_attempt_seq,
                split_lot_id or "-",
                split_batch_id,
                exc,
            )
            return None
    return proposal_rows

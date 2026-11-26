"""Helpers consolidados de liquidacao: deteccao, deduplicacao, WS, log e outcome."""

from __future__ import annotations

import asyncio
from typing import Any

from src.application.services.orchestrator.metrics_utils import neutral_metrics
from src.application.services.orchestrator.settlement_outcome import process_contract_outcome
from src.presentation.terminal.log_context import (
    bind_log_context,
    format_log_context_prefix,
    get_log_context,
    restore_log_context,
)


def contract_payload_is_settled(c: dict) -> bool:
    """Retorna True quando o payload indica contrato encerrado na corretora."""
    status = (c.get("status") or "").upper()
    if status in ("OPEN", "ACTIVE"):
        return False
    if int(c.get("is_settled") or 0) == 1:
        return True
    if int(c.get("is_expired") or 0) == 1:
        return True
    return status in ("WON", "LOST", "EXPIRED", "SOLD", "CLOSED")


def log_cluster_summary(orch: Any) -> None:
    """Emite resumo de performance do cluster encerrado."""
    orch.logger.debug(
        "[C%04d] BANCA FINAL: $%.2f | ACUMULADO: %dW / %dL",
        orch._last_result_cycle_id,
        orch.state.balance,
        orch._session_wins,
        orch._session_losses,
    )
    orch.logger.debug("")
    orch._cluster_results = []
    orch._last_anchor_metrics = neutral_metrics()


def settlement_lock(orch: Any) -> asyncio.Lock:
    """Serializa fontes concorrentes de liquidacao no mesmo orquestrador."""
    lock = getattr(orch, "_contract_settlement_lock", None)
    if not isinstance(lock, asyncio.Lock):
        lock = asyncio.Lock()
        orch._contract_settlement_lock = lock
    return lock


def settlement_seen(orch: Any) -> set[int]:
    """Retem IDs liquidados durante a sessao, mesmo apos limpeza do cluster."""
    seen = getattr(orch, "_settled_contract_ids", None)
    if not isinstance(seen, set):
        seen = set()
        orch._settled_contract_ids = seen
    seen.update(getattr(getattr(orch, "risk_manager", None), "settled_contract_ids", set()))
    return seen


async def fetch_portfolio(ws: Any, *, timeout: float) -> list[dict]:
    """Retorna contratos abertos do portfolio Deriv."""
    res = await ws.send({"portfolio": 1}, timeout=timeout)
    if not isinstance(res, dict) or "error" in res:
        return []
    portfolio = res.get("portfolio")
    if not isinstance(portfolio, dict):
        return []
    contracts = portfolio.get("contracts")
    if not isinstance(contracts, list):
        return []
    return [row for row in contracts if isinstance(row, dict)]


async def fetch_profit_table(
    ws: Any,
    *,
    limit: int,
    offset: int = 0,
    timeout: float,
) -> list[dict]:
    """Retorna linhas do profit_table Deriv."""
    res = await ws.send(
        {"profit_table": 1, "description": 1, "limit": int(limit), "offset": int(offset)},
        timeout=timeout,
    )
    if not isinstance(res, dict) or "error" in res:
        return []
    table = res.get("profit_table")
    rows = table.get("transactions") if isinstance(table, dict) else None
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def process_settlement_outcome(
    orch: Any,
    c: dict,
    contract: Any,
    c_id: int,
    profit: float,
    *,
    audit_direction: str | None = None,
    audit_raw_prob: float | None = None,
) -> None:
    """Atualiza saldo, risco e estatisticas de uma liquidacao."""
    process_contract_outcome(
        orch,
        c,
        contract,
        c_id,
        profit,
        audit_direction=audit_direction,
        audit_raw_prob=audit_raw_prob,
        log_cluster_summary=log_cluster_summary,
    )


def emit_contract_settlement_log(
    orch: Any, line: str, *, c_id: int, sym: str | None = None, cycle_id: int | None = None
) -> None:
    """Emite linha [RESOLVED] com o contexto exato do ciclo e ativo do contrato."""
    orig_cycle = cycle_id if cycle_id is not None else getattr(orch, "_contract_cycle", {}).get(c_id, None)
    if orig_cycle in (None, 0):
        orig_cycle = getattr(orch, "_last_result_cycle_id", None) or None
    symbol = sym or getattr(getattr(orch, "risk_manager", None), "contract_to_symbol", {}).get(c_id)
    if not symbol or symbol == "UNK":
        active_c = getattr(getattr(orch, "state", None), "active_contracts", {}).get(c_id)
        if active_c and getattr(active_c, "symbol", None):
            symbol = active_c.symbol
    prev_cid, prev_sym = get_log_context()
    bind_log_context(cycle_id=orig_cycle, symbol=symbol)
    try:
        if getattr(orch, "_buffer_result_logs", False):
            prefix = format_log_context_prefix()
            target_line = f"{prefix}{line}" if prefix and not line.startswith("[c") else line
            orch._pending_result_logs.append(target_line)
        else:
            orch.logger.info(line)
    finally:
        restore_log_context(prev_cid, prev_sym)

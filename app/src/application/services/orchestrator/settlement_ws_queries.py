"""Consultas WebSocket Deriv para reconciliacao de contratos."""

from __future__ import annotations

from typing import Any


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

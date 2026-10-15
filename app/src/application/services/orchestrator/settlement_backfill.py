"""Fallback de liquidacao via profit_table e portfolio quando o poll falha."""

from __future__ import annotations

import time
from typing import Any

from src.application.services.orchestrator.settlement_detect import contract_payload_is_settled
from src.application.services.orchestrator.settlement_logic import process_contract_settlement
from src.application.services.orchestrator.settlement_rest import settle_rest_contract_when_due
from src.application.services.orchestrator.settlement_utils import is_transient_broker_error, mark_ws_offline


def _profit_from_row(row: dict) -> float:
    """Extrai lucro liquido de uma linha do profit_table da Deriv."""
    if row.get("profit") is not None:
        return float(row.get("profit") or 0.0)
    sell_p = float(row.get("sell_price") or 0.0)
    buy_p = float(row.get("buy_price") or 0.0)
    if sell_p or buy_p:
        return sell_p - buy_p
    return 0.0


def _extract_underlying_from_row(row: dict) -> str | None:
    """Extrai ativo de linha do profit_table, parseando shortcode se necessario."""
    if row.get("symbol"):
        return str(row["symbol"])
    if row.get("underlying"):
        return str(row["underlying"])
    shortcode = row.get("shortcode")
    if isinstance(shortcode, str) and "_" in shortcode:
        parts = shortcode.split("_")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return shortcode if isinstance(shortcode, str) else None


def settlement_payload_from_profit_row(c_id: int, row: dict) -> dict:
    """Monta payload proposal_open_contract a partir de linha do profit_table."""
    profit = _profit_from_row(row)
    api_status = (row.get("contract_status") or row.get("status") or "").strip()
    if not api_status:
        api_status = "won" if profit > 0 else ("lost" if profit < 0 else "expired")
    return {
        "proposal_open_contract": {
            "contract_id": c_id,
            "audit_source": "profit_table",
            "is_settled": 1,
            "status": api_status,
            "profit": profit,
            "balance_after": row.get("balance_after"),
            "underlying": _extract_underlying_from_row(row),
        }
    }


async def subscribe_open_contract(ws: Any, contract_id: int, *, timeout: float) -> None:
    """Inscreve atualizacoes em tempo real para um contrato aberto."""
    await ws.send(
        {
            "proposal_open_contract": 1,
            "contract_id": int(contract_id),
            "subscribe": 1,
        },
        timeout=timeout,
    )


async def fetch_open_contract(ws: Any, contract_id: int, *, timeout: float, subscribe: bool) -> dict | None:
    """Consulta proposal_open_contract e retorna o dict interno ou None."""
    req: dict[str, Any] = {"proposal_open_contract": 1, "contract_id": int(contract_id)}
    if subscribe:
        req["subscribe"] = 1
    res = await ws.send(req, timeout=timeout)
    if "error" in res:
        return None
    poc = res.get("proposal_open_contract")
    return poc if isinstance(poc, dict) else None


async def _reconcile_expired_contract_via_rest(orch: Any, contract_id: int) -> bool:
    """Fallback para liquidar contrato expirado via saldo REST quando WS nao responde."""
    active = getattr(getattr(orch, "state", None), "active_contracts", None)
    if not isinstance(active, dict):
        return False
    contract = active.get(contract_id)
    if not contract:
        return False
    try:
        expiry = float(getattr(contract, "expiry_time", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    now = time.time()
    if expiry <= 0 or now < expiry + 3.0:
        return False
    auth = getattr(orch, "auth", None)
    if not auth or not hasattr(auth, "rest_client"):
        return False
    try:
        balance_after_buy = float(getattr(orch.state, "balance", 0.0) or 0.0)
        await settle_rest_contract_when_due(orch, contract, balance_after_buy=balance_after_buy)
        return True
    except Exception:
        return False


async def reconcile_single_contract(orch: Any, contract_id: int) -> bool:
    """Tenta liquidar um contrato via poll; retorna True se processou settlement."""
    ex = orch.config.get("orchestrator", {}).get("execution", {})
    timeout = float(ex.get("settlement_request_timeout_seconds", 30.0))
    try:
        poc = await fetch_open_contract(orch.ws, contract_id, timeout=timeout, subscribe=True)
    except Exception as exc:
        if is_transient_broker_error(exc):
            mark_ws_offline(orch.ws)
        raise
    if poc and contract_payload_is_settled(poc):
        await process_contract_settlement(orch, {"proposal_open_contract": poc})
        return True
    if not poc:
        settled = await backfill_contract_from_profit_table(orch, contract_id)
        if settled:
            return True
        return await _reconcile_expired_contract_via_rest(orch, contract_id)
    return False


async def backfill_contract_from_profit_table(orch: Any, contract_id: int) -> bool:
    """Busca contrato encerrado no profit_table quando proposal_open_contract falha."""
    c_id = int(contract_id)
    limit = int(orch.config.get("orchestrator", {}).get("execution", {}).get("settlement_profit_table_limit", 60))
    try:
        res = await orch.ws.send({"profit_table": 1, "description": 1, "limit": limit, "offset": 0}, timeout=30.0)
    except Exception as exc:
        if is_transient_broker_error(exc):
            mark_ws_offline(orch.ws)
        return False
    if "error" in res:
        return False
    table = res.get("profit_table")
    rows = table.get("transactions") if isinstance(table, dict) else None
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("contract_id")
        if row_id is None or int(row_id) != c_id:
            continue
        await process_contract_settlement(orch, settlement_payload_from_profit_row(c_id, row))
        return True
    return False


async def backfill_pending_contracts(orch: Any, contract_ids: list[int]) -> int:
    """Tenta liquidar contratos pendentes via profit_table; retorna quantidade processada."""
    settled = 0
    for c_id in list(contract_ids):
        if await backfill_contract_from_profit_table(orch, int(c_id)):
            settled += 1
    return settled

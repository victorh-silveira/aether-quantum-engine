"""Liquidacao de contratos comprados via REST bulk-purchase (sem WSS OTP)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.application.services.orchestrator.settlement_logic import process_contract_settlement
from src.domain.models.trade import Contract


logger = logging.getLogger("AETH")


def _synthetic_poc(
    *,
    contract_id: int,
    won: bool,
    buy_price: float,
    payout: float,
    contract_type: str | None = None,
) -> dict[str, Any]:
    """Monta proposal_open_contract sintetico a partir do resultado inferido."""
    profit = float(payout) - float(buy_price) if won else -float(buy_price)
    sell_price = float(payout) if won else 0.0
    normalized_type = str(contract_type or "").strip().upper() or None
    return {
        "proposal_open_contract": {
            "contract_id": int(contract_id),
            "audit_source": "inferred_rest",
            "is_sold": 1,
            "status": "won" if won else "lost",
            "profit": profit,
            "sell_price": sell_price,
            "buy_price": float(buy_price),
            "payout": float(payout),
            "contract_type": normalized_type,
        }
    }


async def _balance_after_wait(orch: Any, *, wait_seconds: float) -> float:
    """Aguarda o expiry e devolve o saldo REST da conta ativa."""
    await asyncio.sleep(max(0.5, float(wait_seconds)))
    client = orch.auth.rest_client()
    accounts = await client.list_accounts()
    account_id = str(getattr(orch, "deriv_account_id", "") or "")
    for acc in accounts:
        if account_id and str(acc.account_id) == account_id:
            return float(acc.balance)
    if accounts:
        return float(accounts[0].balance)
    return float(getattr(orch.state, "balance", 0.0) or 0.0)


async def settle_rest_contract_when_due(orch: Any, contract: Contract, *, balance_after_buy: float) -> None:
    """Apos expiry, tenta profit_table oficial ou infere won/lost pelo saldo REST."""
    now = time.time()
    delay = max(1.0, float(contract.expiry_time) - now + 3.0)
    await asyncio.sleep(max(0.5, float(delay)))
    c_id = int(contract.contract_id)
    try:
        ws = getattr(orch, "ws", None)
        if ws is not None and getattr(ws, "is_running", False):
            from src.application.services.orchestrator.settlement_backfill import (  # noqa: PLC0415
                backfill_contract_from_profit_table,
            )

            if await backfill_contract_from_profit_table(orch, c_id):
                logger.info("SETTLE_REST | contract=%s liquidado via profit_table oficial", c_id)
                return
    except Exception as exc:
        logger.debug("SETTLE_REST | profit_table indisponivel contract=%s: %s", c_id, exc)

    try:
        bal = await _balance_after_wait(orch, wait_seconds=0.5)
        buy_price = float(contract.buy_price)
        payout = float(contract.payout)
        won = (bal >= float(balance_after_buy)) or (bal >= float(balance_after_buy) - buy_price + payout * 0.4)
        if hasattr(orch, "state") and getattr(orch.state, "balance", None) is not None:
            orch.state.balance = bal
        raw_direction = contract.direction.value if hasattr(contract.direction, "value") else str(contract.direction)
        payload = _synthetic_poc(
            contract_id=c_id,
            won=won,
            buy_price=buy_price,
            payout=payout,
            contract_type=raw_direction,
        )
        logger.info(
            "SETTLE_REST | contract=%s won=%s bal=%.2f after_buy=%.2f payout=%.2f",
            c_id,
            str(won).lower(),
            bal,
            balance_after_buy,
            payout,
        )
        await process_contract_settlement(orch, payload)
    except Exception as exc:
        logger.warning("SETTLE_REST falhou contract=%s: %s", c_id, exc)


def schedule_rest_contract_settlement(orch: Any, contract: Contract, *, balance_after_buy: float) -> None:
    """Agenda liquidacao REST em background para compra bulk-purchase."""
    task = asyncio.create_task(
        settle_rest_contract_when_due(orch, contract, balance_after_buy=balance_after_buy),
        name=f"settle-rest-{contract.contract_id}",
    )
    pending = getattr(orch, "_rest_settlement_tasks", None)
    if not isinstance(pending, set):
        pending = set()
        orch._rest_settlement_tasks = pending

    def _done(done: asyncio.Task) -> None:
        """Remove a task concluida do conjunto de liquidacoes REST."""
        pending.discard(done)

    pending.add(task)
    task.add_done_callback(_done)

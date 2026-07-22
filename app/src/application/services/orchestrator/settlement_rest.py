"""Liquidacao de contratos comprados via REST bulk-purchase (sem WSS OTP)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.application.services.orchestrator.settlement_logic import process_contract_settlement
from src.domain.models.trade import Contract


logger = logging.getLogger("AETH")


def _synthetic_poc(*, contract_id: int, won: bool, buy_price: float, payout: float) -> dict[str, Any]:
    """Monta proposal_open_contract sintetico a partir do resultado inferido."""
    profit = float(payout) - float(buy_price) if won else -float(buy_price)
    sell_price = float(payout) if won else 0.0
    return {
        "proposal_open_contract": {
            "contract_id": int(contract_id),
            "is_sold": 1,
            "status": "won" if won else "lost",
            "profit": profit,
            "sell_price": sell_price,
            "buy_price": float(buy_price),
            "payout": float(payout),
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
    """Apos expiry, infere won/lost pelo saldo REST e liquida via payload sintetico."""
    now = time.time()
    delay = max(1.0, float(contract.expiry_time) - now + 3.0)
    try:
        bal = await _balance_after_wait(orch, wait_seconds=delay)
        buy_price = float(contract.buy_price)
        payout = float(contract.payout)
        win_target = float(balance_after_buy) + payout * 0.85
        won = bal >= win_target
        payload = _synthetic_poc(
            contract_id=int(contract.contract_id),
            won=won,
            buy_price=buy_price,
            payout=payout,
        )
        logger.info(
            "SETTLE_REST | contract=%s won=%s bal=%.2f after_buy=%.2f payout=%.2f",
            contract.contract_id,
            str(won).lower(),
            bal,
            balance_after_buy,
            payout,
        )
        await process_contract_settlement(orch, payload)
    except Exception as exc:
        logger.warning("SETTLE_REST falhou contract=%s: %s", contract.contract_id, exc)


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

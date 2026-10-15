"""Consulta de cotacao Rise/Fall sem executar compra."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection
from src.domain.risk.payout_observation import contract_profit_rate
from src.infrastructure.handlers.trade_handler_requests import build_proposal_request


async def fetch_proposal_snapshot(
    handler: Any, symbol: str, direction: TradeDirection, stake: float, params: dict | None = None
) -> dict | None:
    """Retorna preco, payout, taxa e ID validos da cotacao observada."""
    config = params if params is not None else handler.config.get("risk_management", {}).get("params", {})
    request = build_proposal_request(symbol, direction, stake, config)
    try:
        response = await handler.ws.send(request, timeout=int(handler.ws.request_timeout))
        if not isinstance(response, dict) or "error" in response:
            return None
        proposal = response.get("proposal")
        if not isinstance(proposal, dict):
            return None
        price = float(proposal["ask_price"])
        payout = float(proposal["payout"])
        rate = contract_profit_rate(payout, price)
        if rate is None:
            return None
        handler.latest_payout_rate = rate
        return {"proposal_id": str(proposal.get("id") or ""), "price": price, "payout": payout, "rate": rate}
    except Exception:
        return None

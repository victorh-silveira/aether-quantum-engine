"""Calculo de taxa de payout observado e consulta de cotacao Rise/Fall."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from src.domain.models.trade import TradeDirection
from src.infrastructure.handlers.trade_handler_requests import build_proposal_request


def contract_profit_rate(payout: float, buy_price: float) -> float | None:
    """Calcula taxa liquida de retorno sobre o preco pago."""
    if not (math.isfinite(payout) and math.isfinite(buy_price)) or buy_price <= 0.0 or payout <= buy_price:
        return None
    return (payout - buy_price) / buy_price


def _parse_quote_snapshot(handler: Any, response: Any) -> dict[str, Any] | None:
    """Extrai snapshot estruturado de uma resposta de proposal."""
    proposal = response.get("proposal") if isinstance(response, dict) else None
    if not isinstance(proposal, dict):
        return None
    price = float(proposal["ask_price"])
    payout = float(proposal["payout"])
    rate = contract_profit_rate(payout, price)
    if rate is None:
        return None
    handler.latest_payout_rate = rate
    return {
        "proposal_id": str(proposal.get("id") or ""),
        "price": price,
        "payout": payout,
        "rate": rate,
    }


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
        return _parse_quote_snapshot(handler, response)
    except Exception:
        return None


async def resolve_order_proposal(
    handler: Any,
    symbol: str,
    direction: TradeDirection,
    stake: float,
    p_cfg: dict,
    proposal_req: dict,
    timeout: int,
) -> tuple[str, float, float | None, dict]:
    """Envia proposal na Deriv e retorna (proposal_id, ask_price, payout, proposal)."""
    _ = p_cfg
    t0 = time.monotonic()
    prop_resp = await handler.ws.send(proposal_req, timeout=timeout)
    if "error" in prop_resp:
        err = prop_resp["error"]
        raise RuntimeError(f"Erro na proposta: {err.get('message', 'Erro')} (code={err.get('code', 'unknown')})")
    proposal = prop_resp.get("proposal") or {}
    prop_id = str(proposal.get("id") or "")
    ask_price = float(proposal.get("ask_price") or stake)
    raw_payout = proposal.get("payout")
    payout_val = float(raw_payout) if raw_payout is not None else None
    logging.getLogger("AETH").debug(
        "PROPOSAL | %s %s id=%s ask=%.2f payout=%s dt=%.2fs",
        symbol,
        direction.name,
        prop_id,
        ask_price,
        payout_val,
        time.monotonic() - t0,
    )
    return prop_id, ask_price, payout_val, proposal

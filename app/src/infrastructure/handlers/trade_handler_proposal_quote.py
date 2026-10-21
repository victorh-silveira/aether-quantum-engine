"""Consulta de cotacao Rise/Fall sem executar compra."""

from __future__ import annotations

import time
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


async def resolve_order_proposal(
    handler: Any,
    symbol: str,
    direction: TradeDirection,
    stake: float,
    p_cfg: dict,
    proposal_req: dict,
    timeout: int,
) -> tuple[str, float, float | None, dict]:
    """Retorna prop_id, ask_price, payout e proposal com reaproveitamento de cotacao recente."""
    _ = (symbol, direction)
    cached_id = p_cfg.get("_prequoted_proposal_id")
    cached_ts = float(p_cfg.get("_prequoted_timestamp", 0.0) or 0.0)
    cached_ask = float(p_cfg.get("_prequoted_ask_price", 0.0) or 0.0)
    cached_payout = float(p_cfg.get("_prequoted_payout", 0.0) or 0.0)
    can_use_cache = bool(cached_id and (time.time() - cached_ts) <= 2.5 and abs(cached_ask - float(stake)) < 0.01)
    if can_use_cache:
        if p_cfg.get("_prequoted_rate"):
            handler.latest_payout_rate = float(p_cfg["_prequoted_rate"])
        proposal = {"id": str(cached_id), "ask_price": cached_ask, "payout": cached_payout}
        return str(cached_id), cached_ask, cached_payout, proposal
    proposal_resp = await handler.ws.send(proposal_req, timeout=timeout)
    if "error" in proposal_resp:
        msg = proposal_resp["error"].get("message", "Erro desconhecido")
        raise RuntimeError(f"Erro na proposta: {msg}")
    proposal = proposal_resp.get("proposal")
    if not isinstance(proposal, dict):
        raise RuntimeError("Erro na proposta: resposta sem proposal")
    prop_id = proposal.get("id")
    if not prop_id:
        raise RuntimeError("Erro na proposta: id ausente")
    if "_quote_guard_side_probability" in p_cfg and (
        proposal.get("ask_price") is None or proposal.get("payout") is None
    ):
        raise RuntimeError("Cotacao final Rise/Fall sem preco ou payout; compra bloqueada")
    is_multiplier = str(p_cfg.get("contract_type", "")).upper() == "MULTIPLIER"
    try:
        raw_ask = proposal.get("ask_price")
        ask_price = float(raw_ask if raw_ask is not None else stake)
        raw_payout = proposal.get("payout")
        payout_val = float(raw_payout) if raw_payout is not None else (0.0 if is_multiplier else None)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Cotacao final Rise/Fall sem preco ou payout valido; compra bloqueada") from exc
    return str(prop_id), ask_price, payout_val, proposal

"""Recupera proposta rejeitada com uma compra por parametros no mesmo WSS."""

import logging
import time
from typing import Any


async def recover_unknown_contract_proposal(
    ws: Any,
    response: dict,
    proposal_request: dict,
    stake: float,
    *,
    timeout: int,
    enabled: bool,
    expected_candle_epoch: int,
    period_seconds: int = 300,
    entry_window_seconds: int = 30,
    logger: logging.Logger,
) -> tuple[dict, bool]:
    """Usa buy com parametros uma unica vez apos rejeicao explicita do ID."""
    error = response.get("error")
    message = str(error.get("message", "")) if isinstance(error, dict) else ""
    if not enabled or "unknown contract proposal" not in message.lower():
        return response, False
    now = time.time()
    if (
        period_seconds not in (300, 900, 3600)
        or expected_candle_epoch != int(now // period_seconds) * period_seconds - period_seconds
        or now % period_seconds > entry_window_seconds
    ):
        raise RuntimeError("Erro na compra direta: sinal expirou antes da nova tentativa")
    logger.warning("DERIV: proposal ID rejeitado; tentando compra por parametros no mesmo WSS")
    parameters = {key: value for key, value in proposal_request.items() if key not in ("proposal", "req_id")}
    retry = await ws.send(
        {"buy": "1", "price": round(float(stake), 2), "parameters": parameters},
        timeout=timeout,
    )
    return retry, True

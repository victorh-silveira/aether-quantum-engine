"""Cotacao pareada e compra por ID para Touch/No Touch, sem fallback de contrato."""

import asyncio
import math
import time

from src.domain.math.touch_ticks import candidate_barriers, touch_features
from src.domain.models.touch_policy import TouchPolicy
from src.domain.models.trade import Contract, TradeDirection, TradeStatus


class TouchQuoteRejectedError(ValueError):
    """Rejeicao local comprovadamente anterior ao envio da compra."""


async def quote_touch_candidates(handler, ticks, stake: float, policy: TouchPolicy) -> list[dict]:
    """Cota os dois eventos para cada barreira absoluta usando o mesmo snapshot causal."""
    group_ms = int(ticks[-1][0])
    if not 0 <= time.time() * 1000 - group_ms <= policy.max_quote_age_ms:
        raise ValueError("Ticks Touch desatualizados")

    async def request(barrier, contract_type):
        features = touch_features(ticks, barrier, policy)
        sent_ms = int(time.time() * 1000)
        response = await handler.ws.send(
            {
                "proposal": 1,
                "amount": stake,
                "basis": "stake",
                "currency": "USD",
                "underlying_symbol": policy.symbol,
                "contract_type": contract_type,
                "barrier": f"{barrier:.2f}",
                "duration": policy.duration_seconds,
                "duration_unit": "s",
            },
            timeout=int(handler.ws.request_timeout),
        )
        if response.get("error"):
            handler.logger.info("TOUCH | proposta rejeitada %s: %s", contract_type, response["error"].get("message"))
            return None
        proposal = response["proposal"]
        decision_ms = int(proposal["spot_time"]) * 1000
        if decision_ms < group_ms or abs(sent_ms - decision_ms) > policy.max_quote_age_ms:
            raise ValueError("Timestamp broker ausente, atrasado ou relogio desalinhado")
        ask, payout = float(proposal["ask_price"]), float(proposal["payout"])
        if not math.isfinite(ask + payout) or not 0 < ask <= stake or payout <= ask:
            raise ValueError("Cotacao Touch invalida ou acima do teto")
        return {
            "proposal_id": str(proposal["id"]),
            "group_ms": group_ms,
            "decision_ms": decision_ms,
            "sent_ms": sent_ms,
            "received_ms": int(time.time() * 1000),
            "policy_hash": policy.fingerprint(),
            "symbol": policy.symbol,
            "contract_type": contract_type,
            "barrier": barrier,
            "duration_seconds": policy.duration_seconds,
            "ask_price": ask,
            "payout": payout,
            "features": features,
        }

    results = await asyncio.gather(
        *(request(b, kind) for b in candidate_barriers(ticks, policy) for kind in ("ONETOUCH", "NOTOUCH"))
    )
    return [row for row in results if row is not None]


async def buy_touch_quote(handler, quote: dict, policy: TouchPolicy) -> Contract:
    """Compra exatamente a proposta avaliada; nenhum retry de buy nem substituicao por Rise/Fall."""
    sent_ms = int(time.time() * 1000)
    if str(handler.trading_transport).lower() != "ws":
        raise TouchQuoteRejectedError("Touch exige WSS autenticado e settlement broker; bulk REST nao suportado")
    if (
        quote["policy_hash"] != policy.fingerprint()
        or quote["contract_type"] not in {"ONETOUCH", "NOTOUCH"}
        or not 0 <= sent_ms - quote["decision_ms"] <= policy.max_quote_age_ms
    ):
        raise TouchQuoteRejectedError("Proposta Touch vencida/incompativel")
    response = await handler.ws.send(
        {"buy": quote["proposal_id"], "price": quote["ask_price"]}, timeout=int(handler.ws.request_timeout)
    )
    if response.get("error"):
        raise RuntimeError(f"Compra Touch rejeitada: {response['error'].get('message')}")
    payload = response["buy"]
    payload = {
        **payload,
        "proposal_id": quote["proposal_id"],
        "barrier": quote["barrier"],
        "contract_type": quote["contract_type"],
    }
    direction = TradeDirection[quote["contract_type"]]
    await handler._record_purchase_audit(payload, policy.symbol, direction, sent_ms, int(time.time() * 1000))
    return Contract(
        contract_id=int(payload["contract_id"]),
        proposal_id=quote["proposal_id"],
        status=TradeStatus.OPEN,
        buy_price=float(payload.get("buy_price") or quote["ask_price"]),
        payout=float(payload.get("payout") or quote["payout"]),
        symbol=policy.symbol,
        direction=direction,
        stake=quote["ask_price"],
        expiry_time=int(payload.get("start_time") or sent_ms / 1000) + policy.duration_seconds,
        longcode=str(payload.get("longcode") or ""),
        entry_spot=payload.get("entry_spot"),
        entry_time=payload.get("entry_spot_time"),
    )

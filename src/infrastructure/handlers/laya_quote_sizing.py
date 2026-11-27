"""Revalida saldo, Kelly e amortizacao na proposta efetivamente comprada."""

import logging
import math

from src.domain.risk.kelly_recovery import sizing_from_config
from src.infrastructure.handlers.trade_handler_proposal_quote import contract_profit_rate


async def resize_laya_proposal(handler, symbol, direction, stake, params, request, timeout, resolve):
    """Recota ate tres vezes e compra somente proposta coerente com o risco atual."""
    context = params["_laya_sizing"]
    for _ in range(3):
        response = await handler.ws.send({"balance": 1}, timeout=timeout)
        balance = float(response.get("balance", {}).get("balance", 0))
        if not math.isfinite(balance) or balance <= 0 or "error" in response:
            raise RuntimeError("LAYA_SIZING saldo atual invalido")
        if callable(on_balance := context.get("on_balance")):
            on_balance(balance)
        proposal = await resolve(handler, symbol, direction, stake, {}, request, timeout)
        prop_id, ask, payout, _ = proposal
        rate = contract_profit_rate(payout, ask) if payout is not None else None
        if not prop_id or rate is None:
            raise RuntimeError("LAYA_SIZING proposta sem payout ou ID valido")
        sizing = sizing_from_config(
            balance, float(context["probability"]), rate, float(context["pending"]), context["config"]
        )
        if abs(ask - sizing.stake) < 0.005 and abs(stake - sizing.stake) < 0.005:
            logger = logging.getLogger("AETH")
            logger.info(
                "KELLY | source=%s | base=$%.2f | stake=$%.2f | f=%.2f | payout=%.2f | edge=%.2f | banca=$%.2f",
                sizing.source,
                sizing.base_stake,
                sizing.stake,
                sizing.kelly_raw,
                rate,
                sizing.edge,
                balance,
            )
            logger.info(
                "RECOVERY | pending=$%.2f | parcela=$%.2f | etapas=%d | cap=%s",
                sizing.pending_loss,
                sizing.recovery_slice,
                sizing.recovery_steps,
                sizing.capped,
            )
            return proposal
        stake = sizing.stake
        request = dict(request, amount=stake)
    raise RuntimeError("LAYA_SIZING cotacao instavel; nenhuma compra enviada")

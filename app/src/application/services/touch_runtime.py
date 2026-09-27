"""Fluxo Touch independente de CALL/PUT, reutilizando contabilidade e settlement do motor."""

import asyncio
import math
import time

from src.application.services.orchestrator.engine_supervisor import spawn_background
from src.application.services.orchestrator.settlement_backfill import subscribe_open_contract
from src.application.services.touch_model import select_touch_quote, validate_touch_bundle
from src.domain.models.touch_policy import resolve_touch_policy
from src.infrastructure.handlers.touch_broker import TouchQuoteRejectedError, buy_touch_quote, quote_touch_candidates
from src.infrastructure.market.touch_store import append_touch_quotes, load_touch_model


async def run_touch_cycle(orch) -> bool:
    """Coleta propostas mesmo sem modelo; compra somente com OOS e EV, sem recovery sizing."""
    policy = resolve_touch_policy(orch.config)
    if orch.state.active_contracts or orch.risk_manager.active_contract_ids:
        return False
    if getattr(orch, "_touch_buy_uncertain", False):
        orch.logger.error("TOUCH | compra anterior sem confirmacao; reconciliar antes de reabrir ordens")
        return False
    ticks = orch.stream.tick_buffer.recent_ticks(policy.symbol)
    risk = orch.config["risk_management"]
    balance = float(orch.state.balance)
    cap = min(float(risk["kelly"]["max_stake"]), balance * policy.max_stake_pct)
    minimum = float(risk["params"]["stake_min"])
    if not math.isfinite(cap) or cap < minimum:
        orch.logger.info("TOUCH | saldo insuficiente para stake minimo dentro do teto")
        return False
    try:
        quotes = await quote_touch_candidates(orch.trade_handler, ticks, math.floor(cap * 100) / 100, policy)
        await asyncio.to_thread(append_touch_quotes, quotes)
        bundle = await asyncio.to_thread(load_touch_model)
        validate_touch_bundle(bundle, policy, time.time() * 1000)
        selection = select_touch_quote(bundle, quotes, policy)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        orch.logger.info("TOUCH | coleta/qualificacao pendente: %s", exc)
        return False
    if selection is None:
        orch.logger.info("TOUCH | nenhuma proposta com EV acima de %.3f", policy.min_edge)
        return False
    quote, p_touch, edge = selection
    payout_rate = quote["payout"] / quote["ask_price"] - 1
    kelly_cap = balance * edge / payout_rate * float(risk["kelly"]["kelly_fraction"])
    if quote["ask_price"] > min(cap, kelly_cap):
        orch.logger.info("TOUCH | proposta excede teto Kelly; sem elevar stake ao piso")
        return False
    if str(orch.trade_handler.trading_transport).lower() != "ws":
        orch.logger.warning("TOUCH | dados coletados; compra exige transporte WS autenticado")
        return False
    orch._touch_buy_uncertain = True
    try:
        contract = await buy_touch_quote(orch.trade_handler, quote, policy)
    except TouchQuoteRejectedError as exc:
        orch._touch_buy_uncertain = False
        orch.logger.info("TOUCH | compra nao enviada: %s", exc)
        return False
    cid = contract.contract_id
    orch.risk_manager.record_contract_stake(cid, contract.buy_price)
    orch.risk_manager.active_contract_ids.append(cid)
    orch.risk_manager.contract_to_symbol[cid] = policy.symbol
    await orch.state.add_contract(contract)
    orch._contract_cycle[cid] = int(orch._active_cycle_id)
    orch.risk_manager.begin_cluster(1)
    orch._touch_buy_uncertain = False
    orch.logger.info(
        "TOUCH | EXEC %s barrier=%.2f p_touch=%.4f EV=%+.4f stake=%.2f cid=%s",
        quote["contract_type"],
        quote["barrier"],
        p_touch,
        edge,
        contract.buy_price,
        cid,
    )
    spawn_background(orch, subscribe_open_contract(orch.ws, cid, timeout=30), name=f"touch-settle-{cid}")
    spawn_background(orch, orch.executor._run_settlement_watch(), name=f"touch-watch-{cid}")
    return True

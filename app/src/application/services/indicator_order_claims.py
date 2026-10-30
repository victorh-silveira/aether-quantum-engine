"""Reserva atomica de compra por ativo e vela entre ciclos e processos."""

import logging
import time
from uuid import uuid4

from src.application.services.indicator_purchase_quarantine import purchase_account_key
from src.infrastructure.state.redis_state_store import RedisStateStore


logger = logging.getLogger("AETH")


def claim_indicator_order(orch, symbol: str, metrics: dict) -> bool:
    """Reserva uma compra por ativo e vela, inclusive entre ciclos concorrentes."""
    if metrics.get("indicator_controls_execution") and not metrics.get("indicator_candle_epoch"):
        return False
    cfg = getattr(orch, "config", None)
    cfg_d = cfg if isinstance(cfg, dict) else {}
    period = int(
        metrics.get("indicator_period_seconds") or cfg_d.get("pullback_strategy", {}).get("period_seconds", 300)
    )
    fallback = int(time.time() // period) * period - period
    epoch = int(metrics.get("indicator_candle_epoch") or metrics.get("model_input_candle_epoch") or fallback)
    if epoch <= 0:
        return False
    metrics["_order_candle_epoch"] = epoch
    claims = getattr(orch, "_indicator_order_claims", None)
    if not isinstance(claims, set):
        claims = orch._indicator_order_claims = set()
    claims.intersection_update({item for item in claims if item[2] >= epoch})
    key = (str(symbol), period, epoch)
    if key in claims:
        return False
    claims.add(key)
    return True


def _has_active_contract_for_symbol(orch, symbol: str) -> bool:
    """Verifica se ha contrato em aberto para o simbolo no estado ou risco."""
    state = getattr(orch, "state", None)
    active_contracts = getattr(state, "active_contracts", {})
    if isinstance(active_contracts, dict):
        for c in active_contracts.values():
            if str(getattr(c, "symbol", "") or "") == str(symbol):
                return True
    rm = getattr(orch, "risk_manager", None)
    if rm is not None:
        c_to_sym = getattr(rm, "contract_to_symbol", {})
        active_ids = getattr(rm, "active_contract_ids", set())
        for cid in active_ids:
            if str(c_to_sym.get(cid, "") or "") == str(symbol):
                return True
    return False


async def claim_indicator_order_once(orch, symbol: str, metrics: dict) -> bool:
    """Reserva a vela localmente e no Redis antes de enviar a compra ao broker."""
    blocked = getattr(orch, "_unknown_purchase_until", None)
    if isinstance(blocked, dict) and float(blocked.get(str(symbol), 0.0)) > time.time():
        return False
    store = getattr(orch, "state_store", None)
    client = None
    if isinstance(store, RedisStateStore):
        try:
            client = await store._redis()
            if await client.exists(store._full_key(f"unknown_purchase:{purchase_account_key(orch, symbol)}")):
                return False
        except Exception as exc:
            logger.error("EXEC: estado de compra sem ACK indisponivel para %s: %s", symbol, exc)
            return False
    if not claim_indicator_order(orch, symbol, metrics):
        return False
    if client is None:
        return True
    epoch = int(metrics["_order_candle_epoch"])
    period = int(metrics.get("indicator_period_seconds") or 300)
    key = store._full_key(f"order_claim:{purchase_account_key(orch, symbol)}:{period}:{epoch}")
    token = uuid4().hex
    try:
        claimed = bool(await client.set(key, token, nx=True, ex=900))
        if claimed:
            metrics["_order_claim_token"] = token
            if orch.config.get("indicator_consensus", {}).get("multi_period_trading", False):
                symbol_key = store._full_key(f"order_symbol_claim:{purchase_account_key(orch, symbol)}")
                symbol_claimed = bool(await client.set(symbol_key, token, nx=True, ex=period + 30))
                if not symbol_claimed:
                    if not _has_active_contract_for_symbol(orch, symbol):
                        await client.set(symbol_key, token, ex=period + 30)
                    else:
                        await release_unbought_order_claim(orch, symbol, metrics)
                        return False
                metrics["_symbol_claim_token"] = token
        return claimed
    except Exception as exc:
        logger.error("INDICADOR: reserva Redis indisponivel; compra suspensa para %s: %s", symbol, exc)
        if metrics.get("_order_claim_token"):
            await release_unbought_order_claim(orch, symbol, metrics)
        return False


async def release_order_symbol_claim(orch, symbol: str) -> None:
    """Libera a trava de simbolo no Redis apos liquidacao do contrato."""
    store = getattr(orch, "state_store", None)
    if isinstance(store, RedisStateStore) and symbol:
        symbol_key = store._full_key(f"order_symbol_claim:{purchase_account_key(orch, symbol)}")
        try:
            client = await store._redis()
            await client.delete(symbol_key)
        except Exception as exc:
            logger.debug("EXEC: falha ao liberar trava de simbolo para %s: %s", symbol, exc)


async def release_unbought_order_claim(orch, symbol: str, metrics: dict) -> None:
    """Libera reserva propria apenas quando a proposta terminou sem enviar compra."""
    epoch = int(metrics.get("_order_candle_epoch") or 0)
    period = int(metrics.get("indicator_period_seconds") or 300)
    key = (str(symbol), period, epoch)
    token = metrics.get("_order_claim_token")
    store = getattr(orch, "state_store", None)
    if isinstance(store, RedisStateStore) and token:
        script = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end"
        redis_key = store._full_key(f"order_claim:{purchase_account_key(orch, symbol)}:{period}:{epoch}")
        try:
            client = await store._redis()
            await client.eval(script, 1, redis_key, token)
            if orch.config.get("indicator_consensus", {}).get("multi_period_trading", False):
                symbol_key = store._full_key(f"order_symbol_claim:{purchase_account_key(orch, symbol)}")
                await client.eval(script, 1, symbol_key, token)
        except Exception as exc:
            logger.error("EXEC: falha ao liberar reserva sem compra para %s: %s", symbol, exc)
            return
    claims = getattr(orch, "_indicator_order_claims", None)
    if isinstance(claims, set):
        claims.discard(key)

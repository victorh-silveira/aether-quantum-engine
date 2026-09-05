"""Operacoes de fila de liquidacao sem dependencia de settlement_logic."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import redis.asyncio as aioredis

from src.application.services.orchestrator.engine_supervisor import shield_critical
from src.application.services.orchestrator.settlement_ws_queries import fetch_profit_table
from src.application.services.settle_log_dedupe import log_settle_info_if_changed, log_settle_warning_if_changed
from src.presentation.terminal.settle_log import (
    SETTLE_CONFIRM,
    SETTLE_ENQUEUE,
    SETTLE_ENQUEUE_ERR,
    SETTLE_PROCESS,
    SETTLE_READ,
)


REDIS_SETTLEMENT_QUEUE_KEY = "settlement:queue:priority"
_REDIS_POLL_MIN_INTERVAL_SECONDS = 2.0


def cancel_settlement_queue_fast(orch: Any) -> None:
    """Cancela worker e drena fila sem aguardar handshakes pendentes."""
    worker = getattr(orch, "_settlement_worker_task", None)
    if worker is not None and not worker.done():
        worker.cancel()
    queue = getattr(orch, "_settlement_queue", None)
    if queue is not None:
        while not queue.empty():
            queue.get_nowait()
            queue.task_done()
    orch._settlement_worker_task = None


def _enqueued_ids(orch: Any) -> set[int]:
    """Conjunto local de contract_ids ja enfileirados no Redis nesta sessao."""
    bag = getattr(orch, "_settlement_redis_enqueued", None)
    if not isinstance(bag, set):
        bag = set()
        orch._settlement_redis_enqueued = bag
    return bag


def _redis_lock(orch: Any) -> asyncio.Lock:
    """Lock serializa leituras/escritas da fila de settlement no Redis."""
    lock = getattr(orch, "_settlement_redis_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        orch._settlement_redis_lock = lock
    return lock


async def get_redis_client(orch: Any):
    """Retorna cliente Redis a partir do state_store ou da URL do config."""
    if hasattr(orch, "state_store") and hasattr(orch.state_store, "_redis"):
        return await orch.state_store._redis()
    root = orch.config if isinstance(getattr(orch, "config", None), dict) else {}
    infra = root.get("infra") if isinstance(root.get("infra"), dict) else {}
    redis_cfg = infra.get("redis") if isinstance(infra.get("redis"), dict) else root.get("redis", {})
    if not isinstance(redis_cfg, dict):
        redis_cfg = {}
    redis_url = str(redis_cfg.get("url", "redis://127.0.0.1:6379/0"))
    connect_timeout = float(redis_cfg.get("socket_connect_timeout_seconds", 2.0))
    socket_timeout = float(redis_cfg.get("socket_timeout_seconds", 15.0))
    return aioredis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=max(0.1, connect_timeout),
        socket_timeout=max(0.1, socket_timeout),
    )


async def push_to_redis_priority_queue(orch: Any, payload: dict) -> None:
    """Insere o payload de liquidacao na fila de prioridade do Redis (idempotente por cid)."""
    poc = payload.get("proposal_open_contract", {}) if isinstance(payload, dict) else {}
    raw_id = poc.get("contract_id", 0) if isinstance(poc, dict) else 0
    try:
        c_id = int(raw_id)
    except (TypeError, ValueError):
        c_id = 0
    if c_id <= 0:
        return
    enqueued = _enqueued_ids(orch)
    try:
        async with _redis_lock(orch):
            if c_id in enqueued:
                return
            client = await get_redis_client(orch)

            async def _push() -> None:
                """ZREM+ZADD atomico sob shield_critical para o contract_id."""
                await client.zremrangebyscore(REDIS_SETTLEMENT_QUEUE_KEY, c_id, c_id)
                await client.zadd(REDIS_SETTLEMENT_QUEUE_KEY, {json.dumps(payload): c_id})

            await shield_critical(_push())
            enqueued.add(c_id)
            orch._settlement_redis_poll_at = 0.0
        log_settle_info_if_changed(
            orch,
            orch.logger,
            SETTLE_ENQUEUE,
            f"settle_redis_push:{c_id}",
            "queued",
            "Contrato %s enfileirado no Redis por instabilidade.",
            c_id,
        )
    except Exception as e:
        log_settle_warning_if_changed(
            orch,
            orch.logger,
            SETTLE_ENQUEUE_ERR,
            f"settle_redis_push_err:{c_id}",
            str(e),
            "Falha ao enfileirar no Redis: %s",
            e,
        )


async def _confirm_queued_item(orch: Any, client: Any, item_str: str) -> None:
    """Confirma um item da fila Redis via open_contract ou profit_table."""
    from src.application.services.orchestrator.settlement_backfill import (  # noqa: PLC0415
        fetch_open_contract,
        settlement_payload_from_profit_row,
    )
    from src.application.services.orchestrator.settlement_logic import (  # noqa: PLC0415
        _process_confirmed_settlement,
        contract_payload_is_settled,
        process_late_settlement_from_payload,
    )

    payload = json.loads(item_str)
    poc = payload.get("proposal_open_contract", {})
    c_id = poc.get("contract_id")
    if c_id is None:
        await client.zrem(REDIS_SETTLEMENT_QUEUE_KEY, item_str)
        return
    c_id = int(c_id)
    confirmed = False
    try:
        confirmed_poc = await fetch_open_contract(orch.ws, c_id, timeout=10.0, subscribe=False)
        if confirmed_poc and contract_payload_is_settled(confirmed_poc):
            contract = await orch.state.finalize_contract(c_id)
            if contract:
                await _process_confirmed_settlement(orch, {"proposal_open_contract": confirmed_poc}, contract)
            else:
                await process_late_settlement_from_payload(orch, confirmed_poc)
            confirmed = True
        else:
            rows = await fetch_profit_table(orch.ws, limit=20, timeout=10.0)
            row = next((r for r in rows if int(r.get("contract_id", 0)) == c_id), None)
            if row:
                confirmed_payload = settlement_payload_from_profit_row(c_id, row)
                poc = confirmed_payload["proposal_open_contract"]
                contract = await orch.state.finalize_contract(c_id)
                if contract:
                    await _process_confirmed_settlement(orch, confirmed_payload, contract)
                else:
                    await process_late_settlement_from_payload(orch, poc)
                confirmed = True
    except Exception as exc:
        log_settle_warning_if_changed(
            orch,
            orch.logger,
            SETTLE_CONFIRM,
            f"settle_confirm:{c_id}",
            str(exc),
            "Falha ao confirmar P&L via API para cid=%d: %s",
            c_id,
            exc,
        )
    if confirmed:
        await client.zrem(REDIS_SETTLEMENT_QUEUE_KEY, item_str)
        _enqueued_ids(orch).discard(c_id)


async def process_redis_settlement_queue(orch: Any, *, force: bool = False) -> None:
    """Consome a fila de prioridade do Redis confirmando cada P&L com a API."""
    if not orch.ws.is_running:
        return
    now = time.monotonic()
    last = float(getattr(orch, "_settlement_redis_poll_at", 0.0) or 0.0)
    if not force and (now - last) < _REDIS_POLL_MIN_INTERVAL_SECONDS:
        return
    orch._settlement_redis_poll_at = now
    try:
        async with _redis_lock(orch):
            client = await get_redis_client(orch)
            items = await client.zrange(REDIS_SETTLEMENT_QUEUE_KEY, 0, -1)
            if not items:
                return
            log_settle_info_if_changed(
                orch,
                orch.logger,
                SETTLE_PROCESS,
                "settle_redis_process",
                str(len(items)),
                "Processando %d item(ns) da fila de prioridade do Redis.",
                len(items),
            )
            for item_str in items:
                await _confirm_queued_item(orch, client, item_str)
    except Exception as e:
        log_settle_warning_if_changed(
            orch,
            orch.logger,
            SETTLE_READ,
            "settle_redis_read",
            str(e),
            "Erro ao ler fila de prioridade do Redis: %s",
            e,
        )

"""Operacoes de fila de liquidacao sem dependencia de settlement_logic."""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis

from src.application.services.orchestrator.settlement_ws_queries import fetch_profit_table


REDIS_SETTLEMENT_QUEUE_KEY = "settlement:queue:priority"


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


async def get_redis_client(orch: Any):
    """Retorna cliente Redis a partir do state_store ou da URL do config."""
    if hasattr(orch, "state_store") and hasattr(orch.state_store, "_redis"):
        return await orch.state_store._redis()
    redis_url = orch.config.get("redis", {}).get("url", "redis://localhost:6379/0")
    return aioredis.from_url(redis_url, decode_responses=True)


async def push_to_redis_priority_queue(orch: Any, payload: dict) -> None:
    """Insere o payload de liquidacao na fila de prioridade do Redis."""
    try:
        client = await get_redis_client(orch)
        c_id = payload.get("proposal_open_contract", {}).get("contract_id", 0)
        await client.zadd(REDIS_SETTLEMENT_QUEUE_KEY, {json.dumps(payload): int(c_id)})
        orch.logger.info("SETTLE: Contrato %s enfileirado no Redis por instabilidade.", c_id)
    except Exception as e:
        orch.logger.error("SETTLE: Falha ao enfileirar no Redis: %s", e)


async def process_redis_settlement_queue(orch: Any) -> None:  # noqa: PLR0912
    """Consome a fila de prioridade do Redis confirmando cada P&L com a API de forma sincrona."""
    if not orch.ws.is_running:
        return
    try:
        client = await get_redis_client(orch)
        items = await client.zrange(REDIS_SETTLEMENT_QUEUE_KEY, 0, -1)
        if not items:
            return

        orch.logger.info("SETTLE: Processando %d item(ns) da fila de prioridade do Redis.", len(items))

        from src.application.services.orchestrator.settlement_backfill import (  # noqa: PLC0415
            fetch_open_contract,
            settlement_payload_from_profit_row,
        )
        from src.application.services.orchestrator.settlement_logic import (  # noqa: PLC0415
            _process_confirmed_settlement,
            contract_payload_is_settled,
            process_late_settlement_from_payload,
        )

        for item_str in items:
            payload = json.loads(item_str)
            poc = payload.get("proposal_open_contract", {})
            c_id = poc.get("contract_id")
            if c_id is None:
                await client.zrem(REDIS_SETTLEMENT_QUEUE_KEY, item_str)
                continue

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
                orch.logger.warning("SETTLE: Falha ao confirmar P&L via API para cid=%d: %s", c_id, exc)

            if confirmed:
                await client.zrem(REDIS_SETTLEMENT_QUEUE_KEY, item_str)
    except Exception as e:
        orch.logger.error("SETTLE: Erro ao ler fila de prioridade do Redis: %s", e)

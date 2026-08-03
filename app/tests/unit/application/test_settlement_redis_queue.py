"""Testes unitários para operações de fila Redis de liquidação."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.settlement_logic import (
    process_contract_settlement,
)
from src.application.services.orchestrator.settlement_queue_ops import (
    process_redis_settlement_queue,
    push_to_redis_priority_queue,
)


@pytest.mark.asyncio
async def test_process_contract_settlement_redis_queue_flow(orch_ready):
    """Verifica o fluxo completo de enfileiramento e consumo via Redis queue."""
    orch = orch_ready
    orch.ws.is_running = False

    mock_redis = AsyncMock()

    with (
        patch(
            "src.application.services.orchestrator.settlement_queue_ops.get_redis_client",
            new=AsyncMock(return_value=mock_redis),
        ),
    ):
        await process_contract_settlement(
            orch,
            {
                "proposal_open_contract": {
                    "status": "won",
                    "is_settled": 1,
                    "contract_id": 999,
                    "profit": 10.0,
                }
            },
        )
        mock_redis.zadd.assert_called_once()

        orch.ws.is_running = True
        mock_redis.zrange.return_value = ['{"proposal_open_contract": {"contract_id": 999, "is_settled": 1}}']

        with (
            patch(
                "src.application.services.orchestrator.settlement_logic.process_late_settlement_from_payload",
                new=AsyncMock(),
            ) as mock_late,
            patch(
                "src.application.services.orchestrator.settlement_backfill.fetch_open_contract",
                new=AsyncMock(return_value={"contract_id": 999, "is_settled": 1, "status": "won", "profit": 10.0}),
            ),
        ):
            await process_redis_settlement_queue(orch)
            mock_late.assert_called_once()
            mock_redis.zrem.assert_called_once()


@pytest.mark.asyncio
async def test_process_redis_settlement_queue_with_active_contract(orch_ready):
    """Consome o item da fila Redis quando o contrato correspondente está ativo localmente."""
    orch = orch_ready
    orch.ws.is_running = True

    mock_redis = AsyncMock()
    mock_redis.zrange.return_value = ['{"proposal_open_contract": {"contract_id": 111, "is_settled": 1}}']

    mock_contract = MagicMock()

    with (
        patch.object(orch.state, "finalize_contract", new=AsyncMock(return_value=mock_contract)),
        patch(
            "src.application.services.orchestrator.settlement_queue_ops.get_redis_client",
            new=AsyncMock(return_value=mock_redis),
        ),
        patch(
            "src.application.services.orchestrator.settlement_logic._process_confirmed_settlement", new=AsyncMock()
        ) as mock_confirmed,
        patch(
            "src.application.services.orchestrator.settlement_backfill.fetch_open_contract",
            new=AsyncMock(return_value={"contract_id": 111, "is_settled": 1, "status": "won", "profit": 10.0}),
        ),
    ):
        await process_redis_settlement_queue(orch)
        mock_confirmed.assert_called_once()
        mock_redis.zrem.assert_called_once()


@pytest.mark.asyncio
async def test_process_redis_settlement_queue_fallback_profit_table(orch_ready):
    """Consome o item da fila consultando a profit table da API do broker como fallback."""
    orch = orch_ready
    orch.ws.is_running = True

    mock_redis = AsyncMock()
    mock_redis.zrange.return_value = ['{"proposal_open_contract": {"contract_id": 222, "is_settled": 1}}']

    with (
        patch(
            "src.application.services.orchestrator.settlement_queue_ops.get_redis_client",
            new=AsyncMock(return_value=mock_redis),
        ),
        patch(
            "src.application.services.orchestrator.settlement_backfill.fetch_open_contract",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "src.application.services.orchestrator.settlement_backfill.settlement_payload_from_profit_row",
            return_value={"proposal_open_contract": {"contract_id": 222, "is_settled": 1}},
        ),
        patch(
            "src.application.services.orchestrator.settlement_queue_ops.fetch_profit_table",
            new=AsyncMock(return_value=[{"contract_id": 222, "profit": 5.0}]),
        ),
        patch(
            "src.application.services.orchestrator.settlement_logic.process_late_settlement_from_payload",
            new=AsyncMock(),
        ) as mock_late,
    ):
        await process_redis_settlement_queue(orch)
        mock_late.assert_called_once()


@pytest.mark.asyncio
async def test_push_to_redis_priority_queue_error_block(orch_ready):
    """Verifica tratamento de exceções ao enfileirar falhas no Redis."""
    orch = orch_ready
    mock_redis = AsyncMock()
    mock_redis.zadd.side_effect = RuntimeError("Redis error")

    with (
        patch(
            "src.application.services.orchestrator.settlement_queue_ops.get_redis_client",
            new=AsyncMock(return_value=mock_redis),
        ),
        patch.object(orch.logger, "warning") as mock_warn,
    ):
        await push_to_redis_priority_queue(orch, {"proposal_open_contract": {"contract_id": 999}})
        mock_warn.assert_called_once()


@pytest.mark.asyncio
async def test_process_redis_settlement_queue_empty(orch_ready):
    """Retorna precocemente se a fila de prioridades do Redis estiver vazia."""
    orch = orch_ready
    orch.ws.is_running = True
    mock_redis = AsyncMock()
    mock_redis.zrange.return_value = []

    with (
        patch(
            "src.application.services.orchestrator.settlement_queue_ops.get_redis_client",
            new=AsyncMock(return_value=mock_redis),
        ),
    ):
        await process_redis_settlement_queue(orch)
        mock_redis.zrange.assert_called_once()


@pytest.mark.asyncio
async def test_process_redis_settlement_queue_invalid_item(orch_ready):
    """Drena da fila e descarta itens inválidos sem identificação de contrato."""
    orch = orch_ready
    orch.ws.is_running = True
    mock_redis = AsyncMock()
    mock_redis.zrange.return_value = ['{"proposal_open_contract": {}}']

    with (
        patch(
            "src.application.services.orchestrator.settlement_queue_ops.get_redis_client",
            new=AsyncMock(return_value=mock_redis),
        ),
    ):
        await process_redis_settlement_queue(orch)
        mock_redis.zrem.assert_called_once()


@pytest.mark.asyncio
async def test_process_redis_settlement_queue_fallback_with_active_contract(orch_ready):
    """Consome a profit table para liquidação com um contrato ativo localmente."""
    orch = orch_ready
    orch.ws.is_running = True

    mock_redis = AsyncMock()
    mock_redis.zrange.return_value = ['{"proposal_open_contract": {"contract_id": 333, "is_settled": 1}}']

    mock_contract = MagicMock()

    with (
        patch.object(orch.state, "finalize_contract", new=AsyncMock(return_value=mock_contract)),
        patch(
            "src.application.services.orchestrator.settlement_queue_ops.get_redis_client",
            new=AsyncMock(return_value=mock_redis),
        ),
        patch(
            "src.application.services.orchestrator.settlement_backfill.fetch_open_contract",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "src.application.services.orchestrator.settlement_backfill.settlement_payload_from_profit_row",
            return_value={"proposal_open_contract": {"contract_id": 333, "is_settled": 1}},
        ),
        patch(
            "src.application.services.orchestrator.settlement_queue_ops.fetch_profit_table",
            new=AsyncMock(return_value=[{"contract_id": 333, "profit": 5.0}]),
        ),
        patch(
            "src.application.services.orchestrator.settlement_logic._process_confirmed_settlement", new=AsyncMock()
        ) as mock_confirmed,
    ):
        await process_redis_settlement_queue(orch)
        mock_confirmed.assert_called_once()
        mock_redis.zrem.assert_called_once()


@pytest.mark.asyncio
async def test_process_redis_settlement_queue_exception_handling(orch_ready):
    """Garante resiliência se houver exceções de conexão ou API durante o consumo."""
    orch = orch_ready
    orch.ws.is_running = True

    mock_redis = AsyncMock()
    mock_redis.zrange.return_value = ['{"proposal_open_contract": {"contract_id": 444, "is_settled": 1}}']

    with (
        patch(
            "src.application.services.orchestrator.settlement_queue_ops.get_redis_client",
            new=AsyncMock(return_value=mock_redis),
        ),
        patch(
            "src.application.services.orchestrator.settlement_backfill.fetch_open_contract",
            new=AsyncMock(side_effect=RuntimeError("API error")),
        ),
        patch.object(orch.logger, "warning") as mock_warn,
    ):
        await process_redis_settlement_queue(orch)
        mock_warn.assert_called_once()


@pytest.mark.asyncio
async def test_get_redis_client_uses_infra_url_and_timeouts():
    from src.application.services.orchestrator.settlement_queue_ops import get_redis_client

    orch = MagicMock()
    orch.state_store = MagicMock(spec=[])
    orch.config = {
        "infra": {
            "redis": {
                "url": "redis://127.0.0.1:6379/0",
                "socket_connect_timeout_seconds": 1.25,
                "socket_timeout_seconds": 4.5,
            }
        }
    }
    fake = AsyncMock()
    with patch(
        "src.application.services.orchestrator.settlement_queue_ops.aioredis.from_url",
        return_value=fake,
    ) as from_url:
        client = await get_redis_client(orch)
    assert client is fake
    assert from_url.call_args.args[0] == "redis://127.0.0.1:6379/0"
    assert from_url.call_args.kwargs["socket_connect_timeout"] == pytest.approx(1.25)
    assert from_url.call_args.kwargs["socket_timeout"] == pytest.approx(4.5)


@pytest.mark.asyncio
async def test_push_to_redis_priority_queue_dedupes_same_contract(orch_ready):
    orch = orch_ready
    mock_redis = AsyncMock()
    with patch(
        "src.application.services.orchestrator.settlement_queue_ops.get_redis_client",
        new=AsyncMock(return_value=mock_redis),
    ):
        payload = {"proposal_open_contract": {"contract_id": 777, "is_settled": 1}}
        await push_to_redis_priority_queue(orch, payload)
        await push_to_redis_priority_queue(orch, payload)
        await push_to_redis_priority_queue(orch, {"proposal_open_contract": {"contract_id": "bad"}})
        await push_to_redis_priority_queue(orch, {"proposal_open_contract": {}})
    assert mock_redis.zadd.await_count == 1
    mock_redis.zremrangebyscore.assert_awaited()


@pytest.mark.asyncio
async def test_get_redis_client_ignores_non_dict_redis_block():
    from src.application.services.orchestrator.settlement_queue_ops import get_redis_client

    orch = MagicMock()
    orch.state_store = MagicMock(spec=[])
    orch.config = {"redis": "bad"}
    fake = AsyncMock()
    with patch(
        "src.application.services.orchestrator.settlement_queue_ops.aioredis.from_url",
        return_value=fake,
    ) as from_url:
        assert await get_redis_client(orch) is fake
    assert from_url.call_args.args[0] == "redis://127.0.0.1:6379/0"
    assert from_url.call_args.kwargs["socket_timeout"] == pytest.approx(15.0)

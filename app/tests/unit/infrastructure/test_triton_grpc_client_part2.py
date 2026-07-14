"""Parte 2 dos testes unitarios para cobertura do cliente Triton gRPC."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.infrastructure.inference.triton_grpc_client import (
    TritonGrpcClient,
    _running_loop_or_none,
)


def test_triton_grpc_client_coverage_helpers():
    """Verifica funções auxiliares de detecção e limpeza do loop."""
    assert _running_loop_or_none() is None

    client = TritonGrpcClient()
    assert client.bound_to_running_loop() is False

    client._loop = MagicMock()
    assert client.bound_to_running_loop() is False

    client._infer = MagicMock()
    client._channel = MagicMock()
    client._url = "some"
    client._loop = MagicMock()
    client._lock = MagicMock()
    client.abandon()
    assert client._channel is None
    assert client._loop is None


@pytest.mark.asyncio
async def test_triton_grpc_client_close_stale_loop():
    """Verifica fechamento do cliente quando o event loop mudou."""
    client = TritonGrpcClient()
    client._loop = MagicMock()
    client._infer = MagicMock()
    client.abandon = MagicMock()

    await client.close()
    client.abandon.assert_called_once()


@pytest.mark.asyncio
async def test_triton_grpc_client_close_unlocked_stale_loop():
    """Verifica fechamento destravado quando o loop mudou."""
    client = TritonGrpcClient()
    client._loop = MagicMock()
    client._infer = MagicMock()
    client.abandon = MagicMock()

    await client._close_unlocked()
    client.abandon.assert_called_once()


@pytest.mark.asyncio
async def test_triton_grpc_client_close_runtime_error():
    """Verifica recuperação e limpeza em caso de falha do canal/stub."""
    client = TritonGrpcClient()
    client._loop = asyncio.get_running_loop()
    client._infer = MagicMock()
    client._infer.close = AsyncMock(side_effect=RuntimeError("Mock close error"))
    client.abandon = MagicMock()

    await client.close()
    client.abandon.assert_called_once()

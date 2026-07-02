from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.storage.torchscript_sanity import verify_triton_healthcheck_async


@pytest.mark.asyncio
async def test_verify_triton_healthcheck_skips_when_disabled():
    await verify_triton_healthcheck_async({})


@pytest.mark.asyncio
async def test_verify_triton_healthcheck_async():
    cfg = {"infra": {"triton": {"enabled": True, "grpc_url": "localhost:8001", "http_url": "http://localhost:8000"}}}
    mock_client = MagicMock()
    mock_infer = MagicMock()
    mock_infer.is_server_ready = AsyncMock(return_value=True)
    mock_client._infer = mock_infer
    with (
        patch("src.infrastructure.storage.torchscript_sanity.asyncio.to_thread", new=AsyncMock()),
        patch(
            "src.infrastructure.storage.torchscript_sanity.get_triton_grpc_client",
            AsyncMock(return_value=mock_client),
        ),
    ):
        await verify_triton_healthcheck_async(cfg)
    mock_infer.is_server_ready.assert_awaited_once()


@pytest.mark.asyncio
async def test_verify_triton_healthcheck_http_json_error():
    cfg = {"infra": {"triton": {"enabled": True, "grpc_url": "localhost:8001", "http_url": "http://localhost:8000"}}}
    with (
        patch(
            "src.infrastructure.storage.torchscript_sanity.asyncio.to_thread",
            new=AsyncMock(side_effect=RuntimeError("Triton health/ready: down")),
        ),
        pytest.raises(RuntimeError, match="health/ready"),
    ):
        await verify_triton_healthcheck_async(cfg)


@pytest.mark.asyncio
async def test_verify_triton_healthcheck_http_error_payload():
    cfg = {"infra": {"triton": {"enabled": True, "grpc_url": "localhost:8001", "http_url": "http://localhost:8000"}}}
    with (
        patch(
            "src.infrastructure.storage.torchscript_sanity.asyncio.to_thread",
            new=AsyncMock(side_effect=OSError("connection refused")),
        ),
        pytest.raises(OSError),
    ):
        await verify_triton_healthcheck_async(cfg)


@pytest.mark.asyncio
async def test_verify_triton_healthcheck_grpc_unavailable():
    cfg = {"infra": {"triton": {"enabled": True, "grpc_url": "localhost:8001", "http_url": "http://localhost:8000"}}}
    mock_client = MagicMock()
    mock_client._infer = None
    with (
        patch("src.infrastructure.storage.torchscript_sanity.asyncio.to_thread", new=AsyncMock()),
        patch(
            "src.infrastructure.storage.torchscript_sanity.get_triton_grpc_client",
            AsyncMock(return_value=mock_client),
        ),
        pytest.raises(RuntimeError, match="gRPC indisponivel"),
    ):
        await verify_triton_healthcheck_async(cfg)


@pytest.mark.asyncio
async def test_verify_triton_healthcheck_no_ready_fn():
    cfg = {"infra": {"triton": {"enabled": True, "grpc_url": "localhost:8001", "http_url": "http://localhost:8000"}}}
    mock_client = MagicMock()
    mock_infer = MagicMock(spec=[])
    mock_client._infer = mock_infer
    with (
        patch("src.infrastructure.storage.torchscript_sanity.asyncio.to_thread", new=AsyncMock()),
        patch(
            "src.infrastructure.storage.torchscript_sanity.get_triton_grpc_client",
            AsyncMock(return_value=mock_client),
        ),
    ):
        await verify_triton_healthcheck_async(cfg)


@pytest.mark.asyncio
async def test_verify_triton_healthcheck_not_ready():
    cfg = {"infra": {"triton": {"enabled": True, "grpc_url": "localhost:8001", "http_url": "http://localhost:8000"}}}
    mock_client = MagicMock()
    mock_infer = MagicMock()
    mock_infer.is_server_ready = AsyncMock(return_value=False)
    mock_client._infer = mock_infer
    with (
        patch("src.infrastructure.storage.torchscript_sanity.asyncio.to_thread", new=AsyncMock()),
        patch(
            "src.infrastructure.storage.torchscript_sanity.get_triton_grpc_client",
            AsyncMock(return_value=mock_client),
        ),
        pytest.raises(RuntimeError, match="nao ready"),
    ):
        await verify_triton_healthcheck_async(cfg)

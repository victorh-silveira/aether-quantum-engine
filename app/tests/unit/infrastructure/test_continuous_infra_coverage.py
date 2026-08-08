import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.infrastructure.inference.triton_grpc_client import InferenceServerException
from src.infrastructure.inference.triton_inference_client import (
    close_triton_client,
    get_triton_client,
    infer_symbol_async,
    triton_enabled,
    triton_grpc_url,
)
from src.infrastructure.inference.triton_model_sync import (
    default_triton_repo_path,
    sync_all_symbols_to_triton,
    sync_symbol_torchscript_to_triton,
    triton_config_pbtxt,
)
from src.infrastructure.market.timescale_correlation_reader import (
    correlation_matrix_from_cache,
    correlation_matrix_to_cache,
    fetch_correlation_matrix,
    fetch_symbol_closes,
    read_cached_correlation_matrix,
)
from src.infrastructure.market.timescale_correlation_worker import (
    _correlation_worker_loop,
    refresh_correlation_cache,
)


def test_triton_enabled_and_url():
    assert triton_enabled({"infra": {"triton": {"enabled": True}}})
    assert triton_grpc_url({"infra": {"triton": {"grpc_url": "x:1"}}}) == "x:1"


@pytest.mark.asyncio
async def test_triton_client_pool():
    mock_client = MagicMock()
    with (
        patch(
            "src.infrastructure.inference.triton_inference_client.get_triton_grpc_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ),
        patch(
            "src.infrastructure.inference.triton_inference_client.close_triton_grpc_client",
            new_callable=AsyncMock,
        ),
    ):
        client = await get_triton_client({"infra": {"triton": {"grpc_url": "localhost:8001"}}})
        assert client is mock_client
        await close_triton_client()


@pytest.mark.asyncio
async def test_infer_symbol_async_success():
    fake_client = MagicMock()
    fake_client.infer_symbol = AsyncMock(return_value=0.44)
    with patch(
        "src.infrastructure.inference.triton_inference_client.get_triton_grpc_client",
        new_callable=AsyncMock,
        return_value=fake_client,
    ):
        prob = await infer_symbol_async(
            {"infra": {"triton": {"enabled": True}}},
            "R_10",
            np.zeros((1, 4, 34), dtype=np.float32),
        )
    assert prob == pytest.approx(0.44)


def test_triton_config_and_repo_path():
    text = triton_config_pbtxt(lookback=48, feature_dim=34)
    assert "pytorch_libtorch" in text
    assert 'backend: "pytorch"' in text
    assert default_triton_repo_path().name == "triton-models"


@pytest.mark.asyncio
async def test_sync_all_symbols_to_triton_no_store():
    orch = MagicMock()
    orch.config = {"deep_learning": {}, "data_handler": {}, "risk_management": {"params": {}}}
    orch.symbols = []
    orch.model_store = None
    await sync_all_symbols_to_triton(orch)


@pytest.mark.asyncio
async def test_sync_symbol_downloads_when_missing(tmp_path):
    class Store:
        async def download_torchscript(self, symbol, *, arch, dest):
            _ = (symbol, arch)
            dest.write_bytes(b"x")
            return True

    ok, changed = await sync_symbol_torchscript_to_triton(
        Store(),
        "R_10",
        arch="tcn",
        local_ts_path=tmp_path / "missing.pt",
        lookback=48,
        repo_path_override=tmp_path / "repo",
    )
    assert ok is True
    assert changed is True


def test_correlation_cache_roundtrip():
    matrix = {("R_10", "R_50"): 0.5, ("R_50", "R_10"): 0.5}
    raw = correlation_matrix_to_cache(matrix)
    assert correlation_matrix_from_cache(raw)[("R_10", "R_50")] == 0.5
    assert correlation_matrix_from_cache(None) == {}


@pytest.mark.asyncio
async def test_read_cached_correlation_matrix():
    store = AsyncMock()
    store.get_string.return_value = json.dumps({"R_10|R_10": 0.4})
    orch = MagicMock()
    orch.state_store = store
    matrix = await read_cached_correlation_matrix(orch)
    assert matrix[("R_10", "R_10")] == 0.4


@pytest.mark.asyncio
async def test_fetch_symbol_closes_and_matrix():
    class _Conn:
        async def fetch(self, *_a, **_k):
            return [{"close": 1.0}, {"close": 2.0}, {"close": 3.0}]

    class _Ctx:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *_):
            return False

    class _Pool:
        def acquire(self):
            return _Ctx()

        async def close(self):
            return None

    with patch(
        "src.infrastructure.market.timescale_correlation_reader.asyncpg.create_pool",
        new_callable=AsyncMock,
        return_value=_Pool(),
    ):
        closes = await fetch_symbol_closes("dsn", ["R_10"], granularity=60, bars=3)
        matrix = await fetch_correlation_matrix("dsn", ["R_10", "R_50"], granularity=60, bars=3)
    assert "R_10" in closes
    assert ("R_10", "R_10") in matrix


@pytest.mark.asyncio
async def test_refresh_correlation_cache_enabled():
    orch = MagicMock()
    orch.infra = MagicMock(enabled=True)
    orch.config = {
        "infra": {"timescale": {"dsn": "postgresql://x"}, "triton": {"correlation_bars": 10}},
        "data_handler": {"granularity": 60},
    }
    orch.symbols = ["R_10", "R_50"]
    orch.state_store = AsyncMock()
    with patch(
        "src.infrastructure.market.timescale_correlation_worker.fetch_correlation_matrix",
        new_callable=AsyncMock,
        return_value={("R_10", "R_10"): 0.2},
    ):
        await refresh_correlation_cache(orch)
    orch.state_store.set_string.assert_awaited()


@pytest.mark.asyncio
async def test_correlation_worker_loop_runs_once():
    orch = MagicMock()
    orch.running = True
    orch.config = {
        "orchestrator": {"cycle_interval_seconds": 0},
        "infra": {"triton": {"correlation_refresh_cycles": 1}},
    }
    with (
        patch(
            "src.infrastructure.market.timescale_correlation_worker.refresh_correlation_cache",
            new_callable=AsyncMock,
        ),
        patch(
            "src.infrastructure.market.timescale_correlation_worker.asyncio.sleep",
            new_callable=AsyncMock,
            side_effect=[None, StopAsyncIteration],
        ),
        pytest.raises(StopAsyncIteration),
    ):
        await _correlation_worker_loop(orch)


@pytest.mark.asyncio
async def test_infer_symbol_async_error():
    fake_client = MagicMock()
    fake_client.infer_symbol = AsyncMock(side_effect=InferenceServerException("boom"))
    with (
        patch(
            "src.infrastructure.inference.triton_inference_client.get_triton_grpc_client",
            new_callable=AsyncMock,
            return_value=fake_client,
        ),
        pytest.raises(InferenceServerException),
    ):
        await infer_symbol_async({"infra": {"triton": {}}}, "R_10", np.zeros((1, 4, 34), dtype=np.float32))


@pytest.mark.asyncio
async def test_sync_all_symbols_with_store(tmp_path):
    orch = MagicMock()
    orch.symbols = ["R_10"]
    orch.model_store = MagicMock()
    orch.config = {
        "deep_learning": {"arch": "tcn", "model_path_template": "data/dl/{symbol}.pth"},
        "data_handler": {},
        "risk_management": {"params": {}},
        "infra": {"triton": {"enabled": True}},
    }
    with (
        patch(
            "src.infrastructure.inference.triton_model_sync.sync_symbol_torchscript_to_triton",
            new_callable=AsyncMock,
            return_value=(True, True),
        ) as mock_one,
        patch(
            "src.infrastructure.inference.triton_model_sync.wait_triton_models_stable",
            new_callable=AsyncMock,
            return_value=(True, True),
        ) as mock_reload,
    ):
        await sync_all_symbols_to_triton(orch, repo_path_override=tmp_path)
    mock_one.assert_awaited_once()
    mock_reload.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_symbol_returns_false_without_file_or_download():
    class Store:
        pass

    ok, changed = await sync_symbol_torchscript_to_triton(
        Store(),
        "R_10",
        arch="tcn",
        local_ts_path=Path("missing.pt"),
        lookback=48,
    )
    assert ok is False
    assert changed is False

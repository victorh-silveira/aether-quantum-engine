import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        "infra": {"timescale": {"dsn": "postgresql://x"}, "correlation": {"correlation_bars": 10}},
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
        "infra": {"correlation": {"correlation_refresh_cycles": 1}},
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

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.infrastructure.market import timescale_correlation_worker as correlation_worker
from src.infrastructure.market.timescale_correlation_reader import (
    _log_returns,
    compute_correlation_matrix,
    correlation_matrix_from_cache,
    read_cached_correlation_matrix,
)
from src.infrastructure.market.timescale_correlation_worker import (
    _correlation_worker_loop,
    refresh_correlation_cache,
    start_correlation_worker,
    stop_correlation_worker,
)


def test_correlation_reader_edge_cases():
    assert compute_correlation_matrix({}) == {}
    assert correlation_matrix_from_cache("{bad") == {}
    assert correlation_matrix_from_cache({"x": 1}) == {}


@pytest.mark.asyncio
async def test_refresh_correlation_handles_exception():
    orch = MagicMock()
    orch.infra = MagicMock(enabled=True)
    orch.config = {
        "infra": {"timescale": {"dsn": "postgresql://x"}, "correlation": {}},
        "data_handler": {},
    }
    orch.symbols = ["R_10"]
    with patch(
        "src.infrastructure.market.timescale_correlation_worker.fetch_correlation_matrix",
        new_callable=AsyncMock,
        side_effect=RuntimeError("db"),
    ):
        await refresh_correlation_cache(orch)


@pytest.mark.asyncio
async def test_read_cached_correlation_without_getter():
    orch = MagicMock()
    orch.state_store = object()
    assert await read_cached_correlation_matrix(orch) == {}


def test_correlation_reader_log_returns_edges():
    assert _log_returns([1.0]).size == 0
    assert _log_returns([0.0, 1.0]).size == 0
    short = compute_correlation_matrix({"R_10": [1.0, 1.01, 1.02], "R_50": [2.0, 2.01]})
    assert short[("R_10", "R_50")] == 0.0
    with patch(
        "src.infrastructure.market.timescale_correlation_reader.np.corrcoef",
        return_value=np.array([[1.0, float("nan")], [float("nan"), 1.0]]),
    ):
        nan_matrix = compute_correlation_matrix({"R_10": [1.0, 1.1, 1.2, 1.3, 1.4], "R_50": [2.0, 2.1, 2.2, 2.3, 2.4]})
    assert nan_matrix[("R_10", "R_50")] == 0.0


def test_correlation_cache_invalid_payload():
    assert correlation_matrix_from_cache('["not","dict"]') == {}
    assert correlation_matrix_from_cache(json.dumps({"badkey": 1.0})) == {}


@pytest.mark.asyncio
async def test_read_cached_correlation_no_store():
    assert await read_cached_correlation_matrix(MagicMock(state_store=None)) == {}


@pytest.mark.asyncio
async def test_refresh_correlation_infra_disabled():
    orch = MagicMock()
    orch.infra = None
    await refresh_correlation_cache(orch)


@pytest.mark.asyncio
async def test_correlation_worker_start_skips_when_running():
    orch = MagicMock(running=True)
    orch.config = {"orchestrator": {"cycle_interval_seconds": 60}, "infra": {"correlation": {}}}
    loop = asyncio.get_event_loop()
    correlation_worker._state.task = loop.create_task(asyncio.sleep(9999))
    start_correlation_worker(orch)
    correlation_worker._state.task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await correlation_worker._state.task
    correlation_worker._state.task = None


@pytest.mark.asyncio
async def test_correlation_worker_loop_stops_when_not_running():
    orch = MagicMock()
    orch.running = False
    orch.config = {
        "orchestrator": {"cycle_interval_seconds": 0},
        "infra": {"correlation": {"correlation_refresh_cycles": 2}},
    }
    with patch(
        "src.infrastructure.market.timescale_correlation_worker.refresh_correlation_cache",
        new_callable=AsyncMock,
    ):
        await _correlation_worker_loop(orch)


@pytest.mark.asyncio
async def test_correlation_worker_loop_stops_mid_sleep():
    orch = MagicMock()
    orch.running = True
    orch.config = {
        "orchestrator": {"cycle_interval_seconds": 0},
        "infra": {"correlation": {"correlation_refresh_cycles": 3}},
    }

    async def _sleep(_):
        orch.running = False

    with (
        patch(
            "src.infrastructure.market.timescale_correlation_worker.refresh_correlation_cache",
            new_callable=AsyncMock,
        ),
        patch(
            "src.infrastructure.market.timescale_correlation_worker.asyncio.sleep",
            new_callable=AsyncMock,
            side_effect=_sleep,
        ),
    ):
        await _correlation_worker_loop(orch)


def test_start_correlation_worker_no_running_loop():
    start_correlation_worker(MagicMock())


def test_worker_refresh_without_symbols():
    orch = MagicMock()
    orch.infra = MagicMock(enabled=True)
    orch.config = {"infra": {"timescale": {}, "correlation": {}}, "data_handler": {}}
    orch.symbols = []

    async def _run():
        await refresh_correlation_cache(orch)

    asyncio.run(_run())


def test_stop_correlation_worker_cancels_task():
    loop = asyncio.new_event_loop()
    correlation_worker._state.task = loop.create_task(asyncio.sleep(9999))
    stop_correlation_worker()
    assert correlation_worker._state.task is None
    loop.close()

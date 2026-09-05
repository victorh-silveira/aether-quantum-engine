"""Contrato ensure_timescale / seed policy: pisos micro/macro meta-ready."""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from scripts.operations.timescale_seed_policy import (
    min_bars_for_granularity,
    resolve_seed_bars_for_granularity,
)


def test_min_bars_micro_m5_is_5000():
    assert min_bars_for_granularity(300) == 5000
    assert min_bars_for_granularity(60) == 5000


def test_min_bars_macro_d1_is_365():
    assert min_bars_for_granularity(86400) == 365
    assert min_bars_for_granularity(90000) == 365


def test_resolve_seed_bars_micro_respects_cap():
    assert resolve_seed_bars_for_granularity(300, bars_cap=5000) == 5000
    assert resolve_seed_bars_for_granularity(300, bars_cap=1000) == 5000


def test_resolve_seed_bars_d1_never_asks_5000():
    assert resolve_seed_bars_for_granularity(86400, bars_cap=5000) == 365
    assert resolve_seed_bars_for_granularity(86400, bars_cap=None) == 365


def test_ensure_seed_timeout_returns_one_without_crash():
    from scripts.operations import ensure_timescale as mod

    with patch.object(
        mod.subprocess,
        "run",
        side_effect=mod.subprocess.TimeoutExpired(cmd=["x"], timeout=900),
    ):
        assert mod._seed_timescale(["1HZ75V"]) == 1


@pytest.mark.asyncio
async def test_persist_bundles_uses_executemany():
    from scripts.operations.train_meta_data import OhlcBundle, persist_bundles_to_timescale

    bundle = OhlcBundle(
        symbol="1HZ75V",
        granularity=300,
        closes=np.array([1.0, 1.1], dtype=np.float64),
        open_=np.array([1.0, 1.0], dtype=np.float64),
        high=np.array([1.1, 1.2], dtype=np.float64),
        low=np.array([0.9, 1.0], dtype=np.float64),
        epochs=np.array([1_700_000_000, 1_700_000_300], dtype=np.int64),
        source="test",
    )
    conn = AsyncMock()
    conn.executemany = AsyncMock()
    conn.close = AsyncMock()

    async def _connect(*_a, **_k):
        return conn

    with patch("scripts.operations.train_meta_data.asyncpg.connect", new=_connect):
        written = await persist_bundles_to_timescale("postgresql://x", [bundle])
    assert written == 2
    conn.executemany.assert_awaited_once()
    sql, records = conn.executemany.await_args.args
    assert "ON CONFLICT DO NOTHING" in sql
    assert len(records) == 2

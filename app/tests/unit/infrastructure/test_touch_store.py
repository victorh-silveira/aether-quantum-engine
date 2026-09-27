"""Persistencia isolada de propostas e artefatos Touch."""

from unittest.mock import AsyncMock, patch

import pytest

from src.infrastructure.market import touch_store


def test_journal_and_atomic_model(tmp_path):
    with patch.object(touch_store, "repo_path", side_effect=tmp_path.joinpath):
        touch_store.append_touch_quotes([{"proposal_id": "a"}, {"proposal_id": "b"}])
        assert len(touch_store.read_touch_quotes()) == 2
        path = touch_store.save_touch_model({"qualified": False})
        assert path.is_file() and not path.with_suffix(".tmp").exists()
        assert touch_store.load_touch_model() == {"qualified": False}


@pytest.mark.asyncio
async def test_tick_query_is_parametrized_and_always_closes():
    conn = AsyncMock()
    conn.fetch.return_value = [{"epoch_ms": 1, "price": 100.0}]
    with patch.object(touch_store.asyncpg, "connect", AsyncMock(return_value=conn)):
        assert await touch_store.read_touch_ticks("dsn", "1HZ75V", 1, 2) == [(1, 100.0)]
        assert conn.fetch.await_args.args[1:] == ("1HZ75V", 1, 2)
        conn.fetch.side_effect = RuntimeError("db")
        with pytest.raises(RuntimeError):
            await touch_store.read_touch_ticks("dsn", "1HZ75V", 1, 2)
        assert conn.close.await_count == 2

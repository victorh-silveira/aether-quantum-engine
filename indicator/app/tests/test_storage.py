"""Persistencia SQL do catalogo, sinal e direcao observada."""

import pytest

from indicator.domain import LiveObservation, Signal
from indicator.storage import SignalStore


class Context:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *args):
        return None


class Connection:
    def __init__(self):
        self.calls = []

    async def execute(self, statement, *values):
        self.calls.append((statement, values))

    async def fetch(self, statement, *values):
        self.calls.append((statement, values))
        return [
            {
                "symbol": "R_75",
                "period_seconds": 300,
                "candle_epoch": 600,
                "side": "CALL",
                "probability": 0.6,
                "model_version": "v1",
                "reason": "ok",
                "close": 100,
            }
        ]

    def transaction(self):
        return Context(self)


class Pool:
    def __init__(self):
        self.connection = Connection()
        self.closed = False

    def acquire(self):
        return Context(self.connection)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_store_lifecycle_catalog_and_settlement(monkeypatch):
    pool = Pool()

    async def create_pool(*args, **kwargs):
        assert args == ("postgresql://local",)
        return pool

    monkeypatch.setattr("indicator.storage.asyncpg.create_pool", create_pool)
    store = SignalStore("postgresql://local")
    await store.start()
    await store.set_symbols({"R_75": "V75"})
    await store.replace_models({("R_75", 300): "v1"})
    await store.replace_models({})
    signal = Signal("R_75", 300, 600, "CALL", 0.6, "v1", "ok", 100)
    await store.save(signal, "PUT")
    await store.save(signal)
    await store.save_live(LiveObservation("R_75", 300, 600, 960, 101, 1, "CALL"))
    assert await store.latest_signals() == [signal]
    statements = [item[0] for item in pool.connection.calls]
    assert any("CREATE TABLE" in item for item in statements)
    assert any("UPDATE indicator_signals" in item for item in statements)
    assert sum("INSERT INTO indicator_signals" in item for item in statements) == 2
    assert sum("DELETE FROM indicator_model_versions" in item for item in statements) == 2
    assert any("INSERT INTO indicator_model_versions" in item for item in statements)
    assert any("INSERT INTO indicator_live" in item for item in statements)
    await store.close()


@pytest.mark.asyncio
async def test_store_closes_pool_when_schema_initialization_fails(monkeypatch):
    pool = Pool()

    async def broken(statement, *values):
        raise RuntimeError("schema failed")

    pool.connection.execute = broken

    async def create_pool(*args, **kwargs):
        return pool

    monkeypatch.setattr("indicator.storage.asyncpg.create_pool", create_pool)
    store = SignalStore("postgresql://local")
    with pytest.raises(RuntimeError, match="schema failed"):
        await store.start()
    assert pool.closed
    assert store.pool is None
    assert pool.closed
    await store.close()

"""Persistencia SQL do catalogo, sinal e direcao observada."""

import pytest

from indicator.domain import Signal
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
    signal = Signal("R_75", 300, 600, "CALL", 0.6, "v1", "ok", 100)
    await store.save(signal, "PUT")
    await store.save(signal)
    statements = [item[0] for item in pool.connection.calls]
    assert any("CREATE TABLE" in item for item in statements)
    assert any("UPDATE indicator_signals" in item for item in statements)
    assert sum("INSERT INTO indicator_signals" in item for item in statements) == 2
    await store.close()
    assert pool.closed
    await store.close()

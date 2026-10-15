"""Persistencia SQL do catalogo, sinal e direcao observada."""

import pytest

from indicator.domain import Candle, LiveObservation, Signal
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

    async def executemany(self, statement, values):
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
    await store.save_candles("R_75", 900, [Candle(900, 100, 102, 99, 101)])
    await store.register_model("R_75", 900, "v1", "piloto")
    await store.promote_models()
    await store.save_candidate(signal)
    assert await store.latest_signals() == [signal]
    statements = [item[0] for item in pool.connection.calls]
    assert any("CREATE TABLE" in item for item in statements)
    assert any("UPDATE indicator_signals" in item for item in statements)
    assert sum("INSERT INTO indicator_signals" in item for item in statements) == 2
    assert sum("DELETE FROM indicator_model_versions" in item for item in statements) == 2
    assert any("INSERT INTO indicator_model_versions" in item for item in statements)
    assert any("INSERT INTO indicator_live" in item for item in statements)
    assert any("INSERT INTO indicator_candles" in item for item in statements)
    assert any("INSERT INTO indicator_model_registry" in item for item in statements)
    assert any("INSERT INTO indicator_candidate_signals" in item for item in statements)
    assert any("UPDATE indicator_model_registry" in item for item in statements)
    await store.close()


@pytest.mark.asyncio
async def test_trade_models_selects_one_version_per_state():
    pool = Pool()

    async def rows(statement, *values):
        assert "DISTINCT ON (symbol, period_seconds, state)" in statement
        return [
            {"symbol": "R_25", "period_seconds": 300, "state": "ativo", "model_version": "old"},
            {"symbol": "R_25", "period_seconds": 300, "state": "piloto", "model_version": "new"},
        ]

    pool.connection.fetch = rows
    store = SignalStore("postgresql://local")
    store.pool = pool
    assert await store.trade_models() == {
        ("R_25", 300, "ativo"): "old",
        ("R_25", 300, "piloto"): "new",
    }


@pytest.mark.asyncio
async def test_candle_store_rejects_unknown_period():
    store = SignalStore("postgresql://local")
    with pytest.raises(ValueError, match="periodo nao suportado"):
        await store.save_candles("R_25", 60, [])
    with pytest.raises(ValueError, match="estado inicial invalido"):
        await store.register_model("R_25", 300, "v1", "ativo")


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


@pytest.mark.asyncio
async def test_retained_broker_versions_only_returns_current_versions():
    pool = Pool()

    async def pilot_rows(statement, *values):
        assert "LEFT JOIN indicator_broker_ev_status" in statement
        assert "FROM indicator_model_versions" in statement
        assert "r.period_seconds=m.period_seconds" in statement
        assert "coalesce(e.purchased,0) < 200" in statement
        return [{"symbol": "R_25", "period_seconds": 900, "model_version": "pilot-v1"}]

    pool.connection.fetch = pilot_rows
    store = SignalStore("postgresql://local")
    store.pool = pool
    assert await store.retained_broker_versions() == {("R_25", 900): "pilot-v1"}

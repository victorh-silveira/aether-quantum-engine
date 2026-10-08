"""Fluxo publico de sinal, resultado e falhas."""

import asyncio

import pytest

from indicator.domain import Candle
from indicator.metrics import Metrics
from indicator.model import Model, save
from indicator.service import IndicatorService
from indicator.training import history, train_all


class Source:
    def __init__(self, symbols=None, rows=None):
        self.catalog = symbols if symbols is not None else [{"market": "synthetic_index", "underlying_symbol": "R_75"}]
        self.rows = rows if rows is not None else []
        self.fail = False

    async def symbols(self):
        return self.catalog

    async def candles(self, symbol, count=1000, end="latest"):
        if self.fail:
            raise OSError("offline")
        return self.rows


class Sink:
    def __init__(self):
        self.catalog = None
        self.saved = []

    async def set_symbols(self, symbols):
        self.catalog = symbols

    async def save(self, signal, observed_side=None):
        self.saved.append((signal, observed_side))


def candles(n=60):
    return [
        {"epoch": index * 300, "open": 100 + index, "high": 101 + index, "low": 99 + index, "close": 100 + index}
        for index in range(n)
    ]


@pytest.mark.asyncio
async def test_new_asset_no_model_then_valid_signal_and_outcome(tmp_path):
    source, sink, metrics = Source(rows=candles()), Sink(), Metrics()
    service = IndicatorService(source, sink, metrics, tmp_path)
    await service.tick(60 * 300)
    assert set(service.symbols) == {"R_75"}
    assert len(sink.saved) == 3
    assert all(signal.side == "SEM SINAL" for signal, _ in sink.saved)
    assert any(signal.reason == "modelo_indisponivel" for signal, _ in sink.saved)
    assert all(observed == "CALL" for _, observed in sink.saved)
    await service.tick(60 * 300)
    assert len(sink.saved) == 3
    model = Model("R_75", 300, (0.0,) * 8, 1.0, (0.0,) * 8, (1.0,) * 8, "v1", 0.6, 0.5, 0.2, 0.25)
    save(model, tmp_path)
    source.rows = candles(61)
    await service.tick(61 * 300)
    assert sink.saved[-1][0].period == 300 or sink.saved[-1][0].period == 3600
    assert any(signal.side == "CALL" and signal.model_version == "v1" for signal, _ in sink.saved)
    assert "aether_indicator_side" in metrics.render()


@pytest.mark.asyncio
async def test_stale_short_history_catalog_refresh_and_isolated_error(tmp_path):
    source, sink, metrics = Source(rows=candles(5)), Sink(), Metrics()
    service = IndicatorService(source, sink, metrics, tmp_path)
    await service.tick(1500)
    assert any(signal.reason == "historico_insuficiente" for signal, _ in sink.saved)
    source.rows = candles(30)
    await service.tick(100000)
    assert any(signal.reason == "dados_atrasados" for signal, _ in sink.saved)
    source.fail = True
    await service.tick(100300)
    source.fail = False
    source.catalog = []
    with pytest.raises(RuntimeError, match="vazio"):
        await service.refresh_catalog(100400)
    source.catalog = [{"market": "synthetic_index", "underlying_symbol": "NEW"}]
    await service.refresh_catalog(100500)
    assert metrics.symbols == {"NEW": "NEW"}
    assert all(key[0] == "NEW" for key in service.last_epoch)


@pytest.mark.asyncio
async def test_history_and_training_report(tmp_path, monkeypatch):
    source = Source(rows=candles(600))
    assert len(await history(source, "R_75", 180000, pages=1)) == 600
    source.rows = []
    assert await history(source, "R_75", 180000) == []
    source.rows = candles(600)
    monkeypatch.setattr("indicator.training.history", fake_history)
    report = await train_all(source, tmp_path, now=180000)
    assert len(report) == 3
    assert all("SEM SINAL" in value or "OK" in value for value in report.values())
    source.catalog = []
    with pytest.raises(RuntimeError, match="vazio"):
        await train_all(source, tmp_path, now=180000)


async def fake_history(source, symbol, now):
    return [Candle(index * 300, 100 + index, 100 + index, 100 + index, 100 + index) for index in range(600)]


@pytest.mark.asyncio
async def test_training_continues_other_symbols_on_failure(tmp_path, monkeypatch):
    source = Source(
        symbols=[
            {"market": "synthetic_index", "underlying_symbol": "A"},
            {"market": "synthetic_index", "underlying_symbol": "B"},
        ]
    )

    async def failing_history(source, symbol, now):
        if symbol == "A":
            raise OSError("offline")
        return await fake_history(source, symbol, now)

    monkeypatch.setattr("indicator.training.history", failing_history)
    report = await train_all(source, tmp_path, now=180000)
    assert report["A:300"].startswith("SEM SINAL")
    assert "B:300" in report


@pytest.mark.asyncio
async def test_run_loop_can_be_cancelled(tmp_path, monkeypatch):
    source, sink, metrics = Source(rows=candles()), Sink(), Metrics()
    service = IndicatorService(source, sink, metrics, tmp_path)
    monkeypatch.setattr("indicator.service.time.time", lambda: 18000)

    async def stop(delay):
        raise asyncio.CancelledError

    monkeypatch.setattr("indicator.service.asyncio.sleep", stop)
    with pytest.raises(asyncio.CancelledError):
        await service.run()


@pytest.mark.asyncio
async def test_run_indicator_starts_and_closes_all_adapters(tmp_path, monkeypatch):
    events = []

    class Public:
        def __init__(self, url):
            events.append(("source", url))

        async def close(self):
            events.append("source_closed")

    class Database:
        def __init__(self, dsn):
            events.append(("database", dsn))

        async def start(self):
            events.append("database_started")

        async def close(self):
            events.append("database_closed")

    class Telemetry:
        async def start(self, port):
            events.append(("metrics_started", port))

        async def close(self):
            events.append("metrics_closed")

    async def run(self):
        events.append("running")
        raise asyncio.CancelledError

    monkeypatch.setattr("indicator.service.PublicDeriv", Public)
    monkeypatch.setattr("indicator.service.SignalStore", Database)
    monkeypatch.setattr("indicator.service.Metrics", Telemetry)
    monkeypatch.setattr("indicator.service.IndicatorService.run", run)
    from indicator.service import run_indicator

    with pytest.raises(asyncio.CancelledError):
        await run_indicator("dsn", tmp_path, "url", 9101)
    assert events == [
        ("source", "url"),
        ("database", "dsn"),
        "database_started",
        ("metrics_started", 9101),
        "running",
        "metrics_closed",
        "source_closed",
        "database_closed",
    ]


@pytest.mark.asyncio
async def test_training_exports_accepted_model(tmp_path, monkeypatch):
    source = Source(rows=candles(600))
    monkeypatch.setattr("indicator.training.history", fake_history)
    model = Model("R_75", 300, (0.0,) * 8, 0.0, (0.0,) * 8, (1.0,) * 8, "v2", 0.6, 0.5, 0.2, 0.25)
    monkeypatch.setattr("indicator.training.train", lambda symbol, period, rows: model)
    result = await train_all(source, tmp_path, now=180000)
    assert result["R_75:300"].startswith("OK v2")
    assert (tmp_path / "R_75_300.json").exists()

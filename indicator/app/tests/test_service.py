"""Fluxo publico de sinal, resultado e falhas."""

import asyncio
import logging

import pytest

from indicator.domain import Candle, Signal
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
        self.live = []
        self.models = {}

    async def set_symbols(self, symbols):
        self.catalog = symbols

    async def replace_models(self, versions):
        self.models = versions

    async def save(self, signal, observed_side=None):
        self.saved.append((signal, observed_side))

    async def save_live(self, observation):
        self.live.append(observation)

    async def latest_signals(self):
        return [signal for signal, _ in self.saved]


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
    assert sink.models == {}
    assert any(signal.reason == "modelo_indisponivel" for signal, _ in sink.saved)
    assert all(observed == "CALL" for _, observed in sink.saved)
    await service.tick(60 * 300)
    assert len(sink.saved) == 3
    model = Model("R_75", 300, (0.0,) * 8, 1.0, (0.0,) * 8, (1.0,) * 8, "v1", 0.6, 0.5, 0.2, 0.25)
    save(model, tmp_path)
    source.rows = candles(61)
    await service.tick(61 * 300)
    assert sink.models == {("R_75", 300): "v1"}
    assert sink.saved[-1][0].period == 300 or sink.saved[-1][0].period == 3600
    assert any(signal.side == "CALL" and signal.model_version == "v1" for signal, _ in sink.saved)
    assert "aether_indicator_side" in metrics.render()
    (tmp_path / "R_75_300.json").unlink()
    source.rows = candles(62)
    await service.tick(62 * 300)
    assert sink.models == {}
    assert any(signal.side == "SEM SINAL" and signal.reason == "modelo_indisponivel" for signal, _ in sink.saved[-3:])


@pytest.mark.asyncio
async def test_experimental_model_is_labeled_in_signal(tmp_path):
    model = Model(
        "R_75",
        300,
        (0.0,) * 8,
        0.0,
        (0.0,) * 8,
        (1.0,) * 8,
        "exp",
        0.5,
        0.5,
        0.26,
        0.25,
        "experimental",
    )
    save(model, tmp_path)
    source, sink, metrics = Source(rows=candles()), Sink(), Metrics()
    service = IndicatorService(source, sink, metrics, tmp_path)
    await service.tick(60 * 300)
    assert sink.models == {("R_75", 300): "exp"}
    signal = next(signal for signal, _ in sink.saved if signal.period == 300)
    assert signal.reason == "experimental"
    assert signal.side == "CALL"


@pytest.mark.asyncio
async def test_live_observation_updates_within_candle_without_rewriting_signal(tmp_path):
    model = Model("R_75", 300, (0.0,) * 8, -1.0, (0.0,) * 8, (1.0,) * 8, "v1", 0.6, 0.5, 0.2, 0.25)
    save(model, tmp_path)
    rows = candles(61)
    source, sink, metrics = Source(rows=rows), Sink(), Metrics()
    service = IndicatorService(source, sink, metrics, tmp_path)
    await service.tick(60 * 300 + 60)
    first = next(signal for signal, _ in sink.saved if signal.period == 300)
    assert first.side == "PUT"
    assert sink.live[-1].interim_side == "CALL"
    count = len(sink.saved)
    source.rows[-1] = {**source.rows[-1], "close": 50}
    await service.tick(60 * 300 + 120)
    assert len(sink.saved) == count
    assert sink.live[-1].interim_side == "PUT"
    assert sink.live[-1].observed_at == 60 * 300 + 120
    assert metrics.live[("R_75", 300)].delta < 0


@pytest.mark.asyncio
async def test_retrained_model_waits_for_next_closed_candle(tmp_path):
    first = Model("R_75", 300, (0.0,) * 8, -1.0, (0.0,) * 8, (1.0,) * 8, "old", 0.6, 0.5, 0.2, 0.25)
    save(first, tmp_path)
    source, sink, metrics = Source(rows=candles()), Sink(), Metrics()
    service = IndicatorService(source, sink, metrics, tmp_path)
    await service.tick(60 * 300)
    count = len(sink.saved)
    updated = Model("R_75", 300, (0.0,) * 8, 1.0, (0.0,) * 8, (1.0,) * 8, "new", 0.6, 0.5, 0.2, 0.25)
    save(updated, tmp_path)
    await service.tick(60 * 300)
    assert len(sink.saved) == count
    assert metrics.signals[("R_75", 300)].reason == "modelo_atualizado_aguardando_fechamento"
    assert ("R_75", 300) not in service.last_signals
    source.rows = candles(61)
    await service.tick(61 * 300)
    assert metrics.signals[("R_75", 300)].model_version == "new"


@pytest.mark.asyncio
async def test_restart_restores_issued_candle_without_rewriting_it(tmp_path):
    model = Model("R_75", 300, (0.0,) * 8, 1.0, (0.0,) * 8, (1.0,) * 8, "new", 0.6, 0.5, 0.2, 0.25)
    save(model, tmp_path)
    source, sink, metrics = Source(rows=candles()), Sink(), Metrics()
    service = IndicatorService(source, sink, metrics, tmp_path)
    old = Signal("R_75", 300, 59 * 300, "PUT", 0.4, "old", "ok", 159)
    service.restore([old])
    await service.tick(60 * 300)
    assert all(signal.period != 300 for signal, _ in sink.saved)
    assert metrics.signals[("R_75", 300)].reason == "modelo_atualizado_aguardando_fechamento"
    source.rows = candles(61)
    await service.tick(61 * 300)
    assert metrics.signals[("R_75", 300)].model_version == "new"


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
    source.catalog = [{"market": "synthetic_index", "underlying_symbol": "1HZ10V"}]
    await service.refresh_catalog(100500)
    assert metrics.symbols == {"1HZ10V": "1HZ10V"}
    assert all(key[0] == "1HZ10V" for key in service.last_epoch)


@pytest.mark.asyncio
async def test_history_and_training_report(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    source = Source(rows=candles(600))
    assert len(await history(source, "R_75", 180000, pages=1)) == 600
    source.rows = []
    assert await history(source, "R_75", 180000) == []
    source.rows = candles(600)
    monkeypatch.setattr("indicator.training.history", fake_history)
    report = await train_all(source, tmp_path, now=180000)
    assert len(report) == 3
    assert all("SEM SINAL" in value or "OK" in value or "EXPERIMENTAL" in value for value in report.values())
    assert "TRAIN resumo: 0/3 pares aprovados; 1 experimentais; 2 sem sinal" in caplog.text
    assert "ultima=1970-01-03T02:00:00+00:00" in caplog.text
    source.catalog = []
    with pytest.raises(RuntimeError, match="vazio"):
        await train_all(source, tmp_path, now=180000)


async def fake_history(source, symbol, now):
    return [Candle(index * 300, 100 + index, 100 + index, 100 + index, 100 + index) for index in range(600)]


@pytest.mark.asyncio
async def test_training_continues_other_symbols_on_failure(tmp_path, monkeypatch):
    source = Source(
        symbols=[
            {"market": "synthetic_index", "underlying_symbol": "R_10"},
            {"market": "synthetic_index", "underlying_symbol": "R_25"},
        ]
    )

    async def failing_history(source, symbol, now):
        if symbol == "R_10":
            raise OSError("offline")
        return await fake_history(source, symbol, now)

    monkeypatch.setattr("indicator.training.history", failing_history)
    report = await train_all(source, tmp_path, now=180000)
    assert report["R_10:300"].startswith("SEM SINAL")
    assert "R_25:300" in report


@pytest.mark.asyncio
async def test_run_loop_can_be_cancelled(tmp_path, monkeypatch):
    source, sink, metrics = Source(rows=candles()), Sink(), Metrics()
    service = IndicatorService(source, sink, metrics, tmp_path)
    monkeypatch.setattr("indicator.service.time.time", lambda: 18000)

    async def stop(delay):
        assert delay == 32
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
            if ("database_retry", 2) not in events:
                events.append(("database_retry", 2))
                raise OSError("starting")
            events.append("database_started")

        async def close(self):
            events.append("database_closed")

        async def latest_signals(self):
            events.append("restored")
            return []

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

    async def sleep(seconds):
        assert seconds == 2

    monkeypatch.setattr("indicator.service.asyncio.sleep", sleep)
    from indicator.service import run_indicator

    with pytest.raises(asyncio.CancelledError):
        await run_indicator("dsn", tmp_path, "url", 9101)
    assert events == [
        ("source", "url"),
        ("database", "dsn"),
        ("database_retry", 2),
        "database_started",
        ("metrics_started", 9101),
        "restored",
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
    from indicator.model import TrainingResult

    monkeypatch.setattr(
        "indicator.training.assess",
        lambda symbol, period, rows: TrainingResult(model, 300, 90, 0.6, 0.5, 0.2, 0.25),
    )
    result = await train_all(source, tmp_path, now=180000)
    assert result["R_75:300"].startswith("OK v2")
    assert (tmp_path / "R_75_300.json").exists()


@pytest.mark.asyncio
async def test_training_rejection_archives_previous_checkpoint(tmp_path, monkeypatch):
    source = Source(rows=candles(600))
    monkeypatch.setattr("indicator.training.history", fake_history)
    model = Model("R_75", 300, (0.0,) * 8, 0.0, (0.0,) * 8, (1.0,) * 8, "old", 0.6, 0.5, 0.2, 0.25)
    from indicator.model import load, save

    save(model, tmp_path)
    report = await train_all(source, tmp_path, now=180000)
    assert report["R_75:300"].startswith("EXPERIMENTAL ")
    assert load(tmp_path, "R_75", 300) is None
    assert load(tmp_path, "R_75", 300, allow_experimental=True) is not None
    assert len(list((tmp_path / "rejected").glob("R_75_300_180000_*.json"))) == 1


@pytest.mark.asyncio
async def test_training_total_deadline_reports_no_signal(tmp_path, monkeypatch):
    source = Source()
    monkeypatch.setattr("indicator.training.TRAIN_TIMEOUT", -1)
    result = await train_all(source, tmp_path, now=180000)
    assert all(value == "SEM SINAL: tempo total do treino esgotado" for value in result.values())


@pytest.mark.asyncio
async def test_training_asset_timeout_continues(tmp_path, monkeypatch):
    source = Source(
        symbols=[
            {"market": "synthetic_index", "underlying_symbol": "R_10"},
            {"market": "synthetic_index", "underlying_symbol": "R_25"},
        ]
    )
    monkeypatch.setattr("indicator.training.ASSET_TIMEOUT", 0)
    result = await train_all(source, tmp_path, now=180000)
    assert len(result) == 6
    assert all(value == "SEM SINAL: tempo limite por ativo" for value in result.values())


@pytest.mark.asyncio
async def test_run_indicator_retries_database_and_closes_on_terminal_failure(tmp_path, monkeypatch):
    events = []

    class Database:
        def __init__(self, dsn):
            pass

        async def start(self):
            events.append("start")
            raise OSError("database unavailable")

        async def close(self):
            events.append("database_closed")

    class Public:
        def __init__(self, url):
            pass

        async def close(self):
            events.append("source_closed")

    class Telemetry:
        async def close(self):
            events.append("metrics_closed")

    async def sleep(seconds):
        events.append(("sleep", seconds))

    monkeypatch.setattr("indicator.service.SignalStore", Database)
    monkeypatch.setattr("indicator.service.PublicDeriv", Public)
    monkeypatch.setattr("indicator.service.Metrics", Telemetry)
    monkeypatch.setattr("indicator.service.START_ATTEMPTS", 2)
    monkeypatch.setattr("indicator.service.asyncio.sleep", sleep)
    from indicator.service import run_indicator

    with pytest.raises(OSError, match="database unavailable"):
        await run_indicator("dsn", tmp_path, "url")
    assert events == ["start", ("sleep", 2), "start", "metrics_closed", "source_closed", "database_closed"]


@pytest.mark.asyncio
async def test_service_symbol_timeout_does_not_block_next_symbol(tmp_path):
    class SlowSource(Source):
        async def candles(self, symbol, count=1000, end="latest"):
            if symbol == "R_10":
                await asyncio.sleep(1)
            return candles(60)

    source = SlowSource(
        symbols=[
            {"market": "synthetic_index", "underlying_symbol": "R_10"},
            {"market": "synthetic_index", "underlying_symbol": "R_25"},
        ]
    )
    service = IndicatorService(source, Sink(), Metrics(), tmp_path)
    from indicator import service as module

    original_timeout = module.ASSET_TIMEOUT
    module.ASSET_TIMEOUT = 0.01
    try:
        await service.tick(60 * 300)
    finally:
        module.ASSET_TIMEOUT = original_timeout
    assert ("R_25", 300) in service.last_epoch


@pytest.mark.asyncio
async def test_approved_m5_assets_are_polled_before_catalog_only_assets(tmp_path):
    order = []

    class OrderedSource(Source):
        async def candles(self, symbol, count=1000, end="latest"):
            order.append(symbol)
            return candles(60)

    source = OrderedSource(
        symbols=[
            {"market": "synthetic_index", "underlying_symbol": "R_10"},
            {"market": "synthetic_index", "underlying_symbol": "R_25"},
        ]
    )
    save(Model("R_25", 300, (0.0,) * 8, 1.0, (0.0,) * 8, (1.0,) * 8, "v1", 0.6, 0.5, 0.2, 0.25), tmp_path)
    await IndicatorService(source, Sink(), Metrics(), tmp_path).tick(60 * 300)
    assert order == ["R_25", "R_10"]

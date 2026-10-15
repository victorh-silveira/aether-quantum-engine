"""Fluxo publico de sinal, resultado e falhas."""

import asyncio
import logging

import pytest

from indicator.domain import Candle, Signal
from indicator.metrics import Metrics
from indicator.model import Model, TrainingResult, load, save
from indicator.service import IndicatorService
from indicator.training import _train_asset, history, train_all


class Source:
    def __init__(self, symbols=None, rows=None):
        self.catalog = symbols if symbols is not None else [{"market": "synthetic_index", "underlying_symbol": "R_75"}]
        self.rows = rows if rows is not None else []
        self.fail = False

    async def symbols(self):
        return self.catalog

    async def candles(self, symbol, count=1000, end="latest", period=300):
        if self.fail:
            raise OSError("offline")
        return self.rows


class Sink:
    def __init__(self):
        self.catalog = None
        self.saved = []
        self.live = []
        self.models = {}
        self.candidates = []
        self.trade = {}

    async def set_symbols(self, symbols):
        self.catalog = symbols

    async def replace_models(self, versions):
        self.models = versions

    async def promote_models(self):
        pass

    async def trade_models(self):
        return self.trade

    async def save_candidate(self, signal):
        self.candidates.append(signal)

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
async def test_pilot_challenger_emits_separate_versioned_signal(tmp_path):
    incumbent = Model("R_75", 300, (0.0,) * 8, 1.0, (0.0,) * 8, (1.0,) * 8, "old", 0.6, 0.5, 0.2, 0.25)
    challenger = Model("R_75", 300, (0.0,) * 8, -1.0, (0.0,) * 8, (1.0,) * 8, "new", 0.6, 0.5, 0.2, 0.25)
    save(incumbent, tmp_path)
    save(challenger, tmp_path / "candidates" / "new")
    source, sink, metrics = Source(rows=candles()), Sink(), Metrics()
    sink.trade = {("R_75", 300, "ativo"): "old", ("R_75", 300, "piloto"): "new"}
    service = IndicatorService(source, sink, metrics, tmp_path)
    await service.tick(60 * 300)
    assert len(sink.candidates) == 1
    assert sink.candidates[0].model_version == "new"
    assert sink.candidates[0].side == "PUT"
    assert next(signal for signal, _ in sink.saved if signal.period == 300).model_version == "old"
    await service.tick(60 * 300)
    assert len(sink.candidates) == 1


@pytest.mark.asyncio
async def test_candidate_requires_own_approved_file_and_fresh_history(tmp_path):
    source, sink, metrics = Source(rows=candles()), Sink(), Metrics()
    sink.trade = {("R_75", 300, "piloto"): "missing"}
    service = IndicatorService(source, sink, metrics, tmp_path)
    await service.tick(60 * 300)
    assert sink.candidates == []
    candidate = Model("R_75", 300, (0.0,) * 8, -1.0, (0.0,) * 8, (1.0,) * 8, "different", 0.6, 0.5, 0.2, 0.25)
    save(candidate, tmp_path / "candidates" / "missing")
    source.rows = candles(61)
    await service.tick(61 * 300)
    assert sink.candidates == []


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
    monkeypatch.setattr("indicator.training.MIN_CONTIGUOUS_BARS", 400)
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


@pytest.mark.asyncio
async def test_history_repairs_missing_direct_period_candle():
    calls = []

    class GappedSource(Source):
        async def candles(self, symbol, count=1000, end="latest", period=300):
            calls.append((count, end, period))
            epochs = [0, 900, 2700] if end == "latest" else [1800]
            return [{"epoch": epoch, "open": 1, "high": 2, "low": 1, "close": 2} for epoch in epochs]

    result = await history(GappedSource(), "R_25", 3600, pages=1, period=900)
    assert [row.epoch for row in result] == [0, 900, 1800, 2700]
    assert calls == [(1000, "latest", 900), (3, 2699, 900)]


@pytest.mark.asyncio
async def test_history_does_not_retry_unbounded_gap():
    source = Source(rows=[{"epoch": epoch, "open": 1, "high": 2, "low": 1, "close": 2} for epoch in (0, 1002 * 300)])
    assert len(await history(source, "R_25", 1003 * 300, pages=1)) == 2


@pytest.mark.asyncio
async def test_training_requires_five_thousand_recent_candles(tmp_path, monkeypatch):
    async def short_history(source, symbol, now, period=300):
        return [Candle(index * period, 1, 2, 1, 2) for index in range(4999)]

    monkeypatch.setattr("indicator.training.history", short_history)
    report = await train_all(Source(), tmp_path, now=5000 * 300)
    assert all("historico_continuo_insuficiente" in value for value in report.values())


@pytest.mark.asyncio
async def test_training_persists_direct_candles(tmp_path, monkeypatch):
    monkeypatch.setattr("indicator.training.MIN_CONTIGUOUS_BARS", 2)

    async def short_history(source, symbol, now, period=300):
        return [Candle(index * period, 1, 2, 1, 2) for index in range(3)]

    monkeypatch.setattr("indicator.training.history", short_history)

    class Store:
        def __init__(self):
            self.saved = []
            self.registered = []

        async def save_candles(self, symbol, period, rows):
            self.saved.append((symbol, period, len(rows)))

        async def register_model(self, symbol, period, version, state):
            self.registered.append((symbol, period, version, state))

    store = Store()
    await train_all(Source(), tmp_path, now=900, candle_store=store)
    assert store.saved == [("R_75", period, 3) for period in (300, 900, 3600)]


@pytest.mark.asyncio
async def test_training_rejects_discordant_direct_ohlc(tmp_path, monkeypatch):
    monkeypatch.setattr("indicator.training.MIN_CONTIGUOUS_BARS", 3)

    async def direct_history(source, symbol, now, period=300):
        if period == 300:
            return [Candle(index * 300, 1, 2, 1, 2) for index in range(9)]
        if period == 900:
            return [Candle(index * 900, 1, 3, 1, 2) for index in range(3)]
        return []

    monkeypatch.setattr("indicator.training.history", direct_history)
    report = await train_all(Source(), tmp_path, now=2700)
    assert report["R_75:900"] == "SEM SINAL: divergencia OHLC M5 (3 velas)"


async def fake_history(source, symbol, now, period=300):
    if period != 300:
        return []
    return [Candle(index * 300, 100 + index, 100 + index, 100 + index, 100 + index) for index in range(600)]


@pytest.mark.asyncio
async def test_training_continues_other_symbols_on_failure(tmp_path, monkeypatch):
    source = Source(
        symbols=[
            {"market": "synthetic_index", "underlying_symbol": "R_10"},
            {"market": "synthetic_index", "underlying_symbol": "R_25"},
        ]
    )

    async def failing_history(source, symbol, now, period=300):
        if symbol == "R_10":
            raise OSError("offline")
        return await fake_history(source, symbol, now, period)

    monkeypatch.setattr("indicator.training.history", failing_history)
    report = await train_all(source, tmp_path, now=180000)
    assert report["R_10:300"].startswith("SEM SINAL")
    assert "R_25:300" in report


@pytest.mark.asyncio
async def test_training_write_failure_isolated_to_one_period(tmp_path, monkeypatch):
    monkeypatch.setattr("indicator.training.MIN_CONTIGUOUS_BARS", 300)
    source = Source(rows=candles(600))

    async def longer_history(source, symbol, now, period=300):
        if period == 3600:
            return []
        count = 1200 if period == 300 else 400
        return [Candle(index * period, 100 + index, 100 + index, 100 + index, 100 + index) for index in range(count)]

    monkeypatch.setattr("indicator.training.history", longer_history)
    monkeypatch.setattr("indicator.training.aggregate", lambda *_: [])
    from indicator.model import save as save_model

    def save_with_locked_m5(model, directory):
        if model.period == 300:
            raise PermissionError("arquivo em uso")
        return save_model(model, directory)

    monkeypatch.setattr("indicator.training.save", save_with_locked_m5)
    report = await train_all(source, tmp_path, now=180000)
    assert report["R_75:300"].startswith("SEM SINAL: falha ao gravar modelo")
    assert report["R_75:900"].startswith("EXPERIMENTAL")
    assert report["R_75:3600"].startswith("SEM SINAL: historico_continuo_insuficiente")


@pytest.mark.asyncio
async def test_retraining_preserves_active_demo_pilot_version(tmp_path, monkeypatch):
    monkeypatch.setattr("indicator.training.MIN_CONTIGUOUS_BARS", 500)
    previous = Model("R_75", 300, (0.0,) * 8, 0.0, (0.0,) * 8, (1.0,) * 8, "pilot-v1", 0.6, 0.5, 0.2, 0.25)
    challenger = Model("R_75", 300, (0.1,) * 8, 0.0, (0.0,) * 8, (1.0,) * 8, "challenger-v2", 0.61, 0.5, 0.19, 0.25)
    save(previous, tmp_path)

    async def enough_history(source, symbol, now, period=300):
        if period != 300:
            return []
        return [Candle(index * 300, 100.0, 101.0, 99.0, 100.0) for index in range(600)]

    monkeypatch.setattr("indicator.training.history", enough_history)
    monkeypatch.setattr(
        "indicator.training.assess",
        lambda symbol, period, candles: (
            TrainingResult(challenger, 500, 100, 0.61, 0.5, 0.19, 0.25)
            if period == 300
            else TrainingResult(None, len(candles))
        ),
    )

    class Store:
        def __init__(self):
            self.registered = []

        async def save_candles(self, symbol, period, rows):
            pass

        async def register_model(self, symbol, period, version, state):
            self.registered.append((symbol, period, version, state))

    store = Store()
    report = {}
    await _train_asset(Source(), tmp_path, 180000, "R_75", report, {("R_75", 300): "pilot-v1"}, store)
    assert report["R_75:300"].startswith("VERSAO MANTIDA pilot-v1")
    assert load(tmp_path, "R_75", 300).version == "pilot-v1"
    assert load(tmp_path / "candidates" / "challenger-v2", "R_75", 300).version == "challenger-v2"
    assert store.registered == [("R_75", 300, "challenger-v2", "piloto")]


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

        async def latest_setups(self):
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

    monkeypatch.setattr("indicator.service.PullbackSource", Public)
    monkeypatch.setattr("indicator.service.PullbackStore", Database)
    monkeypatch.setattr("indicator.service.Metrics", Telemetry)
    monkeypatch.setattr("indicator.service.PullbackService.run", run)

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
    monkeypatch.setattr("indicator.training.MIN_CONTIGUOUS_BARS", 500)
    source = Source(rows=candles(600))
    monkeypatch.setattr("indicator.training.history", fake_history)
    model = Model("R_75", 300, (0.0,) * 8, 0.0, (0.0,) * 8, (1.0,) * 8, "v2", 0.6, 0.5, 0.2, 0.25)
    from indicator.model import TrainingResult

    monkeypatch.setattr(
        "indicator.training.assess",
        lambda symbol, period, rows: TrainingResult(model, 300, 90, 0.6, 0.5, 0.2, 0.25),
    )

    class Store:
        def __init__(self):
            self.registered = []

        async def save_candles(self, symbol, period, rows):
            pass

        async def register_model(self, symbol, period, version, state):
            self.registered.append((symbol, period, version, state))

    store = Store()
    result = await train_all(source, tmp_path, now=180000, candle_store=store)
    assert result["R_75:300"].startswith("OK v2")
    assert (tmp_path / "R_75_300.json").exists()
    assert store.registered == [("R_75", 300, "v2", "piloto")]


@pytest.mark.asyncio
async def test_training_rejection_archives_previous_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("indicator.training.MIN_CONTIGUOUS_BARS", 500)
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
async def test_training_without_new_model_archives_and_removes_old_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("indicator.training.MIN_CONTIGUOUS_BARS", 500)
    source = Source(rows=candles(600))
    monkeypatch.setattr("indicator.training.history", fake_history)
    old = Model("R_75", 300, (0.0,) * 8, 0.0, (0.0,) * 8, (1.0,) * 8, "old", 0.6, 0.5, 0.2, 0.25)
    save(old, tmp_path)
    from indicator.model import TrainingResult

    monkeypatch.setattr("indicator.training.assess", lambda *_: TrainingResult(None, 100))
    report = await train_all(source, tmp_path, now=180000)
    assert report["R_75:300"].startswith("SEM SINAL")
    assert not (tmp_path / "R_75_300.json").exists()
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

    monkeypatch.setattr("indicator.service.PullbackStore", Database)
    monkeypatch.setattr("indicator.service.PullbackSource", Public)
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
        async def candles(self, symbol, count=1000, end="latest", period=300):
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
        async def candles(self, symbol, count=1000, end="latest", period=300):
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
    assert order == ["R_25"] * 3 + ["R_10"] * 3

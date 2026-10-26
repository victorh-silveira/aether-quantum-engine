"""Fluxo do setup sem aprovacoes herdadas do indicador antigo."""

import asyncio

import pytest
from test_pullback import call_bars
from test_storage import Pool

from indicator.domain import Signal
from indicator.pullback import MIN_BARS, VERSION
from indicator.pullback_service import PullbackService
from indicator.pullback_source import PullbackSource
from indicator.pullback_store import PullbackStore
from indicator.pullback_training import diagnose_all


class Source:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else []
        self.fail = False

    async def symbols(self):
        return [{"market": "synthetic_index", "underlying_symbol": "R_25"}]

    async def candles(self, symbol, count=100, period=900):
        if self.fail:
            raise OSError("rede")
        return [
            {"epoch": bar.epoch, "open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close}
            for bar in self.rows
        ]


class Sink:
    def __init__(self):
        self.signals = []
        self.promotions = 0

    async def set_symbols(self, symbols):
        self.symbols = symbols

    async def register_setups(self, symbols):
        self.registered = symbols

    async def promote_setups(self):
        self.promotions += 1

    async def save_setup(self, signal):
        self.signals.append(signal)


class Telemetry:
    def __init__(self):
        self.signals = []

    def set_symbols(self, symbols):
        self.symbols = symbols

    def update(self, signal):
        self.signals.append(signal)


@pytest.mark.asyncio
async def test_service_publishes_once_per_closed_candle_and_isolates_failure(monkeypatch):
    bars = call_bars(900)
    source, sink, metrics = Source(bars), Sink(), Telemetry()
    service = PullbackService(source, sink, metrics)
    service.restore([Signal("R_25", 900, 0, "CALL", None, "old", "ok", 100)])
    await service.tick(MIN_BARS * 900)
    assert sink.symbols == {"R_25": "R_25"}
    assert len(sink.signals) >= 1
    call_sig = [s for s in sink.signals if s.period == 900][0]
    assert call_sig.side == "CALL"
    assert call_sig.model_version == VERSION
    count = len(sink.signals)
    await service.tick(MIN_BARS * 900)
    assert len(sink.signals) == count
    source.fail = True
    await service.tick((MIN_BARS + 1) * 900)
    assert len(sink.signals) == count
    assert sink.promotions == 3
    service.restore([call_sig])
    assert service.last_epoch[("R_25", 900)] == call_sig.candle_epoch

    async def stop(_):
        raise asyncio.CancelledError

    monkeypatch.setattr("indicator.pullback_service.asyncio.sleep", stop)
    with pytest.raises(asyncio.CancelledError):
        await service.run()


@pytest.mark.asyncio
async def test_store_writes_isolated_schema(monkeypatch):
    pool = Pool()

    async def create_pool(*args, **kwargs):
        return pool

    monkeypatch.setattr("indicator.storage.asyncpg.create_pool", create_pool)
    store = PullbackStore("postgresql://local")
    await store.start()
    await store.register_setups({"R_25": "V25"})
    await store.promote_setups()
    signal = Signal("R_25", 300, 600, "CALL", None, VERSION, "setup_confirmado", 100)
    await store.save_setup(signal)

    async def fetch(statement, *args):
        return [
            {
                "symbol": "R_25",
                "period_seconds": 300,
                "candle_epoch": 600,
                "side": "CALL",
                "reason": "setup_confirmado",
                "setup_version": VERSION,
                "close": 100,
            }
        ]

    pool.connection.fetch = fetch
    assert await store.latest_setups() == [signal]
    statements = [statement for statement, _ in pool.connection.calls]
    assert any("CREATE OR REPLACE VIEW pullback_broker_status" in item for item in statements)
    assert any("INSERT INTO pullback_registry" in item for item in statements)
    assert any("UPDATE pullback_registry" in item for item in statements)
    await store.close()


@pytest.mark.asyncio
async def test_training_reports_proxy_without_qualifying_contracts(monkeypatch):
    source, sink = Source(), Sink()
    from indicator.domain import Candle

    bars = [
        Candle(i * 900, 100 + i * 0.1, 100 + i * 0.1 + 0.08, 100 + i * 0.1 - 0.08, 100 + i * 0.1 + 0.04)
        for i in range(5000)
    ]

    async def history(_source, _symbol, _now, period):
        return bars if period == 900 else []

    async def save_candles(symbol, period, candles):
        sink.saved = len(candles)

    sink.save_candles = save_candles
    monkeypatch.setattr("indicator.pullback_training.history", history)
    report = await diagnose_all(source, sink, now=5000 * 900)
    assert report["R_25:900"]["estado"] == "PROXY DE VELAS"
    assert report["R_25:900"]["versao"] == VERSION
    assert report["R_25:3600"]["estado"] == "SEM SINAL"
    assert sink.saved == 5000


@pytest.mark.asyncio
async def test_training_proxy_counts_wins_losses_ties_and_handles_failures(monkeypatch):
    from indicator.domain import Candle
    from indicator.pullback import SetupResult

    source, sink = Source(), Sink()
    bars = [
        Candle(i * 900, 100 + i * 0.1, 100 + i * 0.1 + 0.08, 100 + i * 0.1 - 0.08, 100 + i * 0.1 + 0.04)
        for i in range(5000)
    ]
    base = bars[MIN_BARS - 1].close
    bars[MIN_BARS] = Candle(bars[MIN_BARS].epoch, base + 0.5, base + 0.6, base + 0.2, base + 0.3)
    bars[MIN_BARS + 1] = Candle(bars[MIN_BARS + 1].epoch, base + 0.6, base + 0.7, base + 0.3, base + 0.4)
    bars[MIN_BARS + 2] = Candle(bars[MIN_BARS + 2].epoch, base + 0.4, base + 0.48, base + 0.32, base + 0.4)

    async def history(_source, _symbol, _now, period):
        if period == 3600:
            raise OSError("rede")
        return bars if period == 900 else []

    calls = 0

    def setup(*args, **kwargs):
        nonlocal calls
        calls += 1
        return (
            SetupResult(("CALL", "PUT", "CALL")[calls - 1], "setup_confirmado")
            if calls <= 3
            else SetupResult("SEM SINAL", "padrao_incompleto")
        )

    sink.save_candles = AsyncSave()
    monkeypatch.setattr("indicator.pullback_training.history", history)
    monkeypatch.setattr("indicator.pullback_training.evaluate", setup)
    report = await diagnose_all(source, sink, now=5000 * 900)
    assert report["R_25:900"]["acertos"] == 1
    assert report["R_25:900"]["erros"] == 1
    assert report["R_25:900"]["empates"] == 1
    assert report["R_25:900"]["max_losses_consecutivos"] == 1
    assert "optimal_hours_brt" in report["R_25:900"]
    assert "best_window_brt" in report["R_25:900"]
    assert bars[MIN_BARS].close > bars[MIN_BARS - 1].close
    assert bars[MIN_BARS].close < bars[MIN_BARS].open
    assert bars[MIN_BARS + 1].close > bars[MIN_BARS].close
    assert bars[MIN_BARS + 1].close < bars[MIN_BARS + 1].open
    assert report["R_25:3600"]["motivo"] == "rede"

    async def empty():
        return []

    source.symbols = empty
    with pytest.raises(RuntimeError, match="catalogo sintetico vazio"):
        await diagnose_all(source, sink, now=5000 * 300)


class AsyncSave:
    async def __call__(self, *args):
        return None


@pytest.mark.asyncio
async def test_public_source_rotates_four_independent_connections(monkeypatch):
    created = []

    class Client:
        def __init__(self, url):
            created.append(self)
            self.url = url
            self.calls = 0

        async def symbols(self):
            return [{"underlying_symbol": "R_25"}]

        async def candles(self, symbol, count, end, period):
            self.calls += 1
            return []

        async def close(self):
            self.closed = True

    monkeypatch.setattr("indicator.pullback_source.PublicDeriv", Client)
    source = PullbackSource("url")
    assert len(created) == 4
    assert await source.symbols() == [{"underlying_symbol": "R_25"}]
    for _ in range(5):
        assert await source.candles("R_25") == []
    assert [client.calls for client in created] == [2, 1, 1, 1]
    await source.close()
    assert all(client.closed for client in created)
    assert len(PullbackSource("url", 0).clients) == 1


@pytest.mark.asyncio
async def test_service_empty_catalog_and_no_candles():
    source, sink, metrics = Source(), Sink(), Telemetry()

    async def empty():
        return []

    source.symbols = empty
    with pytest.raises(RuntimeError, match="catalogo sintetico vazio"):
        await PullbackService(source, sink, metrics).tick(1000)
    source = Source()
    service = PullbackService(source, sink, metrics)
    await service.tick(1000)
    assert sink.signals == []
    source.rows = call_bars(900)
    await service.tick(MIN_BARS * 900)
    count = len(sink.signals)
    await service.tick((MIN_BARS + 1) * 900)
    assert len(sink.signals) >= count
    assert sum(signal.period == 900 for signal in sink.signals) == 1


def test_compute_hourly_profile_branches():
    from indicator.pullback_training import compute_hourly_profile

    by_hour = {10: [5, 1, 0], 11: [4, 2, 0], 12: [1, 4, 0]}
    optimal, window, profile = compute_hourly_profile(by_hour)
    assert 10 in optimal
    assert "10:00 as" in window

    fallback_hour = {14: [1, 0, 0]}
    fallback_opt, fallback_win, _ = compute_hourly_profile(fallback_hour)
    assert 14 in fallback_opt

    empty_opt, empty_window, _ = compute_hourly_profile({})
    assert empty_opt == []
    assert "05:00 as 13:00" in empty_window

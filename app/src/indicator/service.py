"""Ciclo de observacao de sintéticos sem rotas de negociacao."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Protocol

import asyncpg

from indicator.deriv import PublicDeriv
from indicator.domain import (
    PERIODS,
    Signal,
    aggregate,
    closed_candles,
    current_spot,
    latest_contiguous,
    live_observation,
    synthetic_symbols,
)
from indicator.metrics import Metrics
from indicator.model import LOOKBACK, load
from indicator.storage import SignalStore

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT = 40
ASSET_TIMEOUT = 45
CATALOG_TIMEOUT = 45
START_ATTEMPTS = 8
POLL_SECONDS = 30


class Source(Protocol):
    """Porta publica de market data."""

    async def symbols(self) -> list[dict[str, object]]: ...

    async def candles(self, symbol: str, count: int = 1000, end: str | int = "latest") -> list[dict[str, object]]: ...


class Sink(Protocol):
    """Porta de historico do indicador."""

    async def set_symbols(self, symbols: dict[str, str]) -> None: ...

    async def replace_models(self, versions: dict[tuple[str, int], str]) -> None: ...

    async def save(self, signal: Signal, observed_side: str | None = None) -> None: ...

    async def save_live(self, observation) -> None: ...

    async def latest_signals(self) -> list[Signal]: ...


class IndicatorService:
    """Coordena catalogo, velas fechadas, modelos e observabilidade."""

    def __init__(self, source: Source, sink: Sink, metrics: Metrics, models: Path) -> None:
        self.source = source
        self.sink = sink
        self.metrics = metrics
        self.models = models
        self.symbols: dict[str, str] = {}
        self.last_catalog = 0.0
        self.last_epoch: dict[tuple[str, int], int] = {}
        self.last_signals: dict[tuple[str, int], Signal] = {}

    def restore(self, signals: list[Signal]) -> None:
        """Recupera emissões persistidas para não recriar a mesma vela."""
        for signal in signals:
            key = (signal.symbol, signal.period)
            self.last_epoch[key] = signal.candle_epoch
            self.last_signals[key] = signal
            self.metrics.update(signal)

    async def refresh_catalog(self, now: float) -> None:
        """Atualiza ativos sem manter instrumentos removidos."""
        symbols = synthetic_symbols(await asyncio.wait_for(self.source.symbols(), CATALOG_TIMEOUT))
        if not symbols:
            raise RuntimeError("catalogo sintetico vazio")
        await self.sink.set_symbols(symbols)
        self.metrics.set_symbols(symbols)
        self.symbols = symbols
        self.last_catalog = now
        self.last_epoch = {key: epoch for key, epoch in self.last_epoch.items() if key[0] in symbols}
        self.last_signals = {key: signal for key, signal in self.last_signals.items() if key[0] in symbols}

    async def observe(self, symbol: str, now: int) -> list[Signal]:
        """Publica uma leitura por vela fechada e periodo."""
        rows = await asyncio.wait_for(self.source.candles(symbol), REQUEST_TIMEOUT)
        base = closed_candles(rows, now)
        spot = current_spot(rows, now)
        result = []
        for period in PERIODS:
            candles = latest_contiguous(aggregate(base, period), period)
            if not candles:
                continue
            latest = candles[-1]
            key = (symbol, period)
            if self.last_epoch.get(key) != latest.epoch:
                reason = "ok"
                side = "SEM SINAL"
                probability = None
                version = None
                if latest.epoch + 2 * period < now:
                    reason = "dados_atrasados"
                elif len(candles) < LOOKBACK + 1:
                    reason = "historico_insuficiente"
                else:
                    model = load(self.models, symbol, period, allow_experimental=True)
                    if model is None:
                        reason = "modelo_indisponivel"
                    else:
                        probability = model.predict(candles)
                        version = model.version
                        side = "CALL" if probability >= 0.5 else "PUT"
                        reason = "ok" if model.status == "aprovado" else "experimental"
                signal = Signal(symbol, period, latest.epoch, side, probability, version, reason, latest.close)
                observed = None
                if len(candles) >= 2:
                    observed = (
                        "CALL"
                        if latest.close > candles[-2].close
                        else "PUT"
                        if latest.close < candles[-2].close
                        else "EMPATE"
                    )
                await self.sink.save(signal, observed)
                self.metrics.update(signal)
                self.last_epoch[key] = latest.epoch
                self.last_signals[key] = signal
                result.append(signal)
            if spot is not None and key in self.last_signals:
                live = live_observation(self.last_signals[key], spot, now)
                if live is not None:
                    await self.sink.save_live(live)
                    self.metrics.update_live(live)
        return result

    async def tick(self, now: int) -> None:
        """Percorre catalogo a cada 30 segundos, isolando falhas de ativo."""
        if now - self.last_catalog >= 21_600 or not self.symbols:
            await self.refresh_catalog(float(now))
        versions = {}
        for symbol in self.symbols:
            for period in PERIODS:
                model = load(self.models, symbol, period, allow_experimental=True)
                if model is not None:
                    versions[(symbol, period)] = model.version
                previous = self.last_signals.get((symbol, period))
                if (
                    previous is not None
                    and previous.model_version is not None
                    and (model is None or previous.model_version != model.version)
                ):
                    self.last_signals.pop((symbol, period))
                    self.metrics.update(
                        Signal(
                            symbol,
                            period,
                            previous.candle_epoch,
                            "SEM SINAL",
                            None,
                            None,
                            "modelo_atualizado_aguardando_fechamento",
                            previous.close,
                        )
                    )
        await self.sink.replace_models(versions)
        for symbol in self.symbols:
            try:
                signals = await asyncio.wait_for(self.observe(symbol, now), ASSET_TIMEOUT)
                for signal in signals:
                    logger.info(
                        "SIGNAL %s %ds %s p=%s reason=%s version=%s",
                        symbol,
                        signal.period,
                        signal.side,
                        signal.probability,
                        signal.reason,
                        signal.model_version,
                    )
            except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
                logger.warning("MARKET_DATA %s indisponivel: %s", symbol, str(exc) or "tempo limite")

    async def run(self) -> None:
        """Atualiza o mercado a cada 30 segundos; sinal confirmado usa vela fechada."""
        while True:
            now = int(time.time())
            await self.tick(now)
            delay = max(1, (now // POLL_SECONDS + 1) * POLL_SECONDS - time.time() + 2)
            await asyncio.sleep(delay)


async def run_indicator(dsn: str, models: Path, url: str, metrics_port: int = 9100) -> None:
    """Inicializa adapters e encerra recursos ordenadamente."""
    source = PublicDeriv(url)
    sink = SignalStore(dsn)
    metrics = Metrics()
    try:
        for attempt in range(1, START_ATTEMPTS + 1):
            try:
                await asyncio.wait_for(sink.start(), timeout=15)
                break
            except (OSError, TimeoutError, asyncpg.PostgresError) as exc:
                if attempt == START_ATTEMPTS:
                    raise
                logger.warning("BANCO indisponivel (%d/%d): %s", attempt, START_ATTEMPTS, exc)
                await asyncio.sleep(min(attempt * 2, 10))
        await metrics.start(port=metrics_port)
        service = IndicatorService(source, sink, metrics, models)
        service.restore(await sink.latest_signals())
        await service.run()
    finally:
        await metrics.close()
        await source.close()
        await sink.close()

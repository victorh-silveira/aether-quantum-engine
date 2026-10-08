"""Ciclo de observacao de sintéticos sem rotas de negociacao."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Protocol

from indicator.deriv import PublicDeriv
from indicator.domain import PERIODS, Signal, aggregate, closed_candles, latest_contiguous, synthetic_symbols
from indicator.metrics import Metrics
from indicator.model import LOOKBACK, load
from indicator.storage import SignalStore

logger = logging.getLogger(__name__)


class Source(Protocol):
    """Porta publica de market data."""

    async def symbols(self) -> list[dict[str, object]]: ...

    async def candles(self, symbol: str, count: int = 1000, end: str | int = "latest") -> list[dict[str, object]]: ...


class Sink(Protocol):
    """Porta de historico do indicador."""

    async def set_symbols(self, symbols: dict[str, str]) -> None: ...

    async def save(self, signal: Signal, observed_side: str | None = None) -> None: ...


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

    async def refresh_catalog(self, now: float) -> None:
        """Atualiza ativos sem manter instrumentos removidos."""
        symbols = synthetic_symbols(await self.source.symbols())
        if not symbols:
            raise RuntimeError("catalogo sintetico vazio")
        await self.sink.set_symbols(symbols)
        self.metrics.set_symbols(symbols)
        self.symbols = symbols
        self.last_catalog = now
        self.last_epoch = {key: epoch for key, epoch in self.last_epoch.items() if key[0] in symbols}

    async def observe(self, symbol: str, now: int) -> list[Signal]:
        """Publica uma leitura por vela fechada e periodo."""
        rows = await self.source.candles(symbol)
        base = closed_candles(rows, now)
        result = []
        for period in PERIODS:
            candles = latest_contiguous(aggregate(base, period), period)
            if not candles:
                continue
            latest = candles[-1]
            key = (symbol, period)
            if self.last_epoch.get(key) == latest.epoch:
                continue
            reason = "ok"
            side = "SEM SINAL"
            probability = None
            version = None
            if latest.epoch + 2 * period < now:
                reason = "dados_atrasados"
            elif len(candles) < LOOKBACK + 1:
                reason = "historico_insuficiente"
            else:
                model = load(self.models, symbol, period)
                if model is None:
                    reason = "modelo_indisponivel"
                else:
                    probability = model.predict(candles)
                    version = model.version
                    side = "CALL" if probability >= 0.5 else "PUT"
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
            result.append(signal)
        return result

    async def tick(self, now: int) -> None:
        """Percorre catalogo a cada abertura M5, isolando falhas de ativo."""
        if now - self.last_catalog >= 21_600 or not self.symbols:
            await self.refresh_catalog(float(now))
        for symbol in self.symbols:
            try:
                signals = await self.observe(symbol, now)
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
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning("MARKET_DATA %s indisponivel: %s", symbol, exc)

    async def run(self) -> None:
        """Alinha o polling a abertura M5 sem abrir contratos."""
        while True:
            now = int(time.time())
            await self.tick(now)
            delay = max(1, (now // 300 + 1) * 300 - time.time() + 2)
            await asyncio.sleep(delay)


async def run_indicator(dsn: str, models: Path, url: str, metrics_port: int = 9100) -> None:
    """Inicializa adapters e encerra recursos ordenadamente."""
    source = PublicDeriv(url)
    sink = SignalStore(dsn)
    metrics = Metrics()
    await sink.start()
    try:
        await metrics.start(port=metrics_port)
        service = IndicatorService(source, sink, metrics, models)
        await service.run()
    finally:
        await metrics.close()
        await source.close()
        await sink.close()

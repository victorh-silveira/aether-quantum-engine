"""Observacao publica dos setups de recuo por ativo e periodo."""

from __future__ import annotations

import asyncio
import logging
import time

from indicator.domain import PERIODS, Signal, closed_candles, synthetic_symbols
from indicator.pullback import evaluate, version_for_variant


logger = logging.getLogger(__name__)


class PullbackService:
    """Publica setups fechados com estado de piloto separado dos modelos antigos."""

    def __init__(self, source, sink, metrics, variant: str = "base", scope: str = "all") -> None:
        self.source = source
        self.sink = sink
        self.metrics = metrics
        self.variant = variant
        self.scope = scope
        self.version = version_for_variant(variant)
        self.symbols: dict[str, str] = {}
        self.last_catalog = 0
        self.last_epoch: dict[tuple[str, int], int] = {}

    def restore(self, signals: list[Signal]) -> None:
        """Restaura apenas sinais da versao atual."""
        for signal in signals:
            if signal.model_version == self.version:
                self.last_epoch[(signal.symbol, signal.period)] = signal.candle_epoch
                self.metrics.update(signal)

    async def tick(self, now: int) -> None:
        """Consulta velas diretas e publica a ultima vela fechada de cada periodo."""
        started_at = time.monotonic()
        try:
            if now - self.last_catalog >= 21600 or not self.symbols:
                symbols = synthetic_symbols(await asyncio.wait_for(self.source.symbols(), 40), scope=self.scope)
                if not symbols:
                    raise RuntimeError("catalogo sintetico vazio")
                await self.sink.set_symbols(symbols)
                if hasattr(self.sink, "register_setups"):
                    try:
                        await self.sink.register_setups(symbols, self.version)
                    except TypeError:
                        await self.sink.register_setups(symbols)
                self.metrics.set_symbols(symbols)
                self.symbols = symbols
                self.last_catalog = now
            try:
                await self.sink.promote_setups()
            except Exception as exc:
                logger.warning("PROMOTE_SETUPS falhou: %s", exc)
            ordered_symbols = sorted(self.symbols, key=lambda s: 0 if s == "OTC_SSMI" else 1)
            tasks_m15 = []
            tasks_macro = []
            for symbol in ordered_symbols:
                for period in PERIODS:
                    key = (symbol, period)
                    expected = now // period * period - period
                    if self.last_epoch.get(key) == expected:
                        continue
                    task = self._observe(symbol, period, now)
                    if period == 900:
                        tasks_m15.append(task)
                    else:
                        tasks_macro.append(task)
            if tasks_m15:
                await asyncio.gather(*tasks_m15, return_exceptions=True)
            if tasks_macro:
                await asyncio.gather(*tasks_macro, return_exceptions=True)
        finally:
            if hasattr(self.metrics, "record_tick_duration"):
                self.metrics.record_tick_duration(time.monotonic() - started_at)

    async def _observe(self, symbol: str, period: int, now: int) -> None:
        """Isola erro de um periodo sem atrasar os demais ativos."""
        try:
            rows = await asyncio.wait_for(self.source.candles(symbol, count=500, period=period), 25)
            candles = closed_candles(rows, now, period)
            if not candles:
                return
            latest = candles[-1]
            key = (symbol, period)
            if self.last_epoch.get(key) == latest.epoch:
                return
            result = evaluate(candles, period, now, variant=self.variant, allow_session_gaps=symbol.startswith("OTC_"))
            signal = Signal(
                symbol, period, latest.epoch, result.side, None, result.version, result.reason, latest.close
            )
            await self.sink.save_setup(signal)
            self.last_epoch[key] = latest.epoch
            self.metrics.update(signal)
            logger.info(
                "SETUP %s %ds %s reason=%s version=%s", symbol, period, result.side, result.reason, result.version
            )
        except Exception as exc:
            logger.warning("SETUP %s %ds indisponivel: %s", symbol, period, exc)

    async def run(self) -> None:
        """Observa o catalogo continuamente com isolamento por periodo."""
        while True:
            now = int(time.time())
            try:
                await self.tick(now)
            except Exception as exc:
                logger.error("INDICATOR_TICK_ERROR: %s", exc)
                await asyncio.sleep(5)
            await asyncio.sleep(max(0.25, (now // 30 + 1) * 30 - time.time() + 0.3))

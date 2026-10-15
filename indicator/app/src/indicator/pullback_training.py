"""Diagnostico historico causal do setup fixo, sem qualificar contratos."""

from __future__ import annotations

import asyncio
import logging
import time

from indicator.domain import PERIODS, latest_contiguous, synthetic_symbols
from indicator.pullback import MIN_BARS, evaluate, version_for_variant
from indicator.pullback_research import compare_variants
from indicator.training import history

logger = logging.getLogger(__name__)


async def diagnose_all(source, store, now: int | None = None, variant: str = "base") -> dict[str, dict]:
    """Coleta velas diretas e mede apenas o proxy de fechamento seguinte."""
    timestamp = int(time.time()) if now is None else now
    symbols = synthetic_symbols(await asyncio.wait_for(source.symbols(), 40))
    if not symbols:
        raise RuntimeError("catalogo sintetico vazio")
    version = version_for_variant(variant)
    await store.set_symbols(symbols)
    try:
        await store.register_setups(symbols, version)
    except TypeError:
        await store.register_setups(symbols)
    report: dict[str, dict] = {}
    for symbol in symbols:
        for period in PERIODS:
            key = f"{symbol}:{period}"
            try:
                candles = latest_contiguous(
                    await asyncio.wait_for(history(source, symbol, timestamp, period=period), 90), period
                )
                if candles:
                    await store.save_candles(symbol, period, candles)
                if len(candles) < 5000 or candles[-1].epoch + 2 * period <= timestamp:
                    report[key] = {
                        "estado": "SEM SINAL",
                        "motivo": "historico_continuo_insuficiente",
                        "velas": len(candles),
                    }
                    continue
                wins = losses = ties = 0
                by_side = {"CALL": [0, 0], "PUT": [0, 0]}
                for index in range(MIN_BARS - 1, len(candles) - 1):
                    result = evaluate(
                        candles[index - MIN_BARS + 1 : index + 1],
                        period,
                        candles[index].epoch + period,
                        variant,
                    )
                    if result.side not in by_side:
                        continue
                    next_candle = candles[index + 1]
                    delta = next_candle.close - next_candle.open
                    won = delta > 0 if result.side == "CALL" else delta < 0
                    if delta == 0:
                        ties += 1
                    elif won:
                        wins += 1
                        by_side[result.side][0] += 1
                    else:
                        losses += 1
                        by_side[result.side][1] += 1
                report[key] = {
                    "estado": "PROXY DE VELAS",
                    "versao": version,
                    "velas": len(candles),
                    "setups": wins + losses + ties,
                    "acertos": wins,
                    "erros": losses,
                    "empates": ties,
                    "por_lado": by_side,
                    "variantes": compare_variants(candles, period) if period == 300 else {},
                }
            except (OSError, RuntimeError, ValueError, KeyError, TimeoutError) as exc:
                report[key] = {"estado": "SEM SINAL", "motivo": str(exc) or "tempo limite"}
            result = report[key]
            logger.info(
                "TREINO SETUP %s %ds estado=%s velas=%d setups=%d acertos_proxy=%d erros_proxy=%d motivo=%s",
                symbol,
                period,
                result["estado"],
                result.get("velas", 0),
                result.get("setups", 0),
                result.get("acertos", 0),
                result.get("erros", 0),
                result.get("motivo", "-"),
            )
    return report

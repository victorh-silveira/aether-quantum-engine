"""Diagnostico historico causal do setup fixo, sem qualificar contratos."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

from indicator.domain import PERIODS, latest_contiguous, synthetic_symbols
from indicator.pullback import MIN_BARS, evaluate, version_for_variant
from indicator.pullback_research import compare_variants
from indicator.training import history

logger = logging.getLogger(__name__)

BRT_TIMEZONE = timezone(timedelta(hours=-3))


def compute_hourly_profile(
    by_hour: dict[int, list[int]], payout: float = 0.85
) -> tuple[list[int], str, dict[int, dict[str, float | int]]]:
    """Determina os melhores horarios e janela ideal de operacao em Horario de Brasilia."""
    profile: dict[int, dict[str, float | int]] = {}
    optimal_hours: list[int] = []
    for hour in range(24):
        stats = by_hour.get(hour, [0, 0, 0])
        wins, losses, ties = stats[0], stats[1], stats[2]
        total = wins + losses
        win_rate = (wins / total) if total > 0 else 0.0
        ev = (win_rate * payout) - ((1.0 - win_rate) * 1.0) if total > 0 else -1.0
        profile[hour] = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "setups": total + ties,
            "win_rate": round(win_rate, 4),
            "ev": round(ev, 4),
        }
        if total >= 2 and ev > 0.0:
            optimal_hours.append(hour)

    if not optimal_hours:
        for hour in range(24):
            if profile[hour]["win_rate"] >= 0.5405 and profile[hour]["setups"] > 0:
                optimal_hours.append(hour)

    if optimal_hours:
        best_window = f"{min(optimal_hours):02d}:00 as {max(optimal_hours) + 1:02d}:00 (BRT)"
    else:
        best_window = "05:00 as 13:00 (BRT)"
    return optimal_hours, best_window, profile


async def diagnose_all(
    source, store, now: int | None = None, variant: str = "base", scope: str = "all"
) -> dict[str, dict]:
    """Coleta velas diretas e mede apenas o proxy de fechamento seguinte."""
    timestamp = int(time.time()) if now is None else now
    symbols = synthetic_symbols(await asyncio.wait_for(source.symbols(), 40), scope=scope)
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
                raw_candles = await asyncio.wait_for(history(source, symbol, timestamp, period=period), 90)
                is_stock = scope == "stock_indices" or symbol.startswith("OTC_")
                candles = raw_candles if is_stock else latest_contiguous(raw_candles, period)
                if candles:
                    await store.save_candles(symbol, period, candles)
                min_candles = 200 if is_stock else 5000
                if len(candles) < min_candles:
                    report[key] = {
                        "estado": "SEM SINAL",
                        "motivo": "historico_insuficiente",
                        "velas": len(candles),
                    }
                    continue
                wins = losses = ties = 0
                current_consecutive_losses = 0
                max_consecutive_losses = 0
                by_side = {"CALL": [0, 0], "PUT": [0, 0]}
                by_hour_brt: dict[int, list[int]] = {}
                for index in range(MIN_BARS - 1, len(candles) - 1):
                    if is_stock and candles[index + 1].epoch - candles[index].epoch > period:
                        continue
                    result = evaluate(
                        candles[index - MIN_BARS + 1 : index + 1],
                        period,
                        candles[index].epoch + period,
                        variant,
                    )
                    if result.side not in by_side:
                        continue
                    next_candle = candles[index + 1]
                    hour_brt = datetime.fromtimestamp(candles[index].epoch, tz=BRT_TIMEZONE).hour
                    if hour_brt not in by_hour_brt:
                        by_hour_brt[hour_brt] = [0, 0, 0]
                    delta = next_candle.close - next_candle.open
                    won = delta > 0 if result.side == "CALL" else delta < 0
                    if delta == 0:
                        ties += 1
                        by_hour_brt[hour_brt][2] += 1
                        current_consecutive_losses = 0
                    elif won:
                        wins += 1
                        by_side[result.side][0] += 1
                        by_hour_brt[hour_brt][0] += 1
                        current_consecutive_losses = 0
                    else:
                        losses += 1
                        by_side[result.side][1] += 1
                        by_hour_brt[hour_brt][1] += 1
                        current_consecutive_losses += 1
                        if current_consecutive_losses > max_consecutive_losses:
                            max_consecutive_losses = current_consecutive_losses
                optimal_hours, best_window, hourly_profile = compute_hourly_profile(by_hour_brt)
                report[key] = {
                    "estado": "PROXY DE VELAS",
                    "versao": version,
                    "velas": len(candles),
                    "setups": wins + losses + ties,
                    "acertos": wins,
                    "erros": losses,
                    "empates": ties,
                    "max_losses_consecutivos": max_consecutive_losses,
                    "por_lado": by_side,
                    "optimal_hours_brt": optimal_hours,
                    "best_window_brt": best_window,
                    "hourly_profile_brt": hourly_profile,
                    "variantes": compare_variants(candles, period) if period == 300 else {},
                }
            except (OSError, RuntimeError, ValueError, KeyError, TimeoutError) as exc:
                report[key] = {"estado": "SEM SINAL", "motivo": str(exc) or "tempo limite"}
            result = report[key]
            logger.info(
                "TREINO SETUP %s %ds estado=%s velas=%d setups=%d acertos_proxy=%d erros_proxy=%d max_losses=%d motivo=%s",
                symbol,
                period,
                result["estado"],
                result.get("velas", 0),
                result.get("setups", 0),
                result.get("acertos", 0),
                result.get("erros", 0),
                result.get("max_losses_consecutivos", 0),
                result.get("motivo", "-"),
            )
    return report

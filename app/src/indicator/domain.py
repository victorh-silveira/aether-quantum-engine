"""Tipos e transformacoes puras do indicador de sinteticos."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

PERIODS = (300, 900, 3600)
PERIOD_NAMES = {300: "M5", 900: "M15", 3600: "H1"}


@dataclass(frozen=True)
class Candle:
    """Vela OHLC fechada ou em formacao."""

    epoch: int
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Signal:
    """Leitura de um ativo e periodo, sem instrucao de compra."""

    symbol: str
    period: int
    candle_epoch: int
    side: str
    probability: float | None
    model_version: str | None
    reason: str
    close: float


def synthetic_symbols(rows: list[dict[str, object]]) -> dict[str, str]:
    """Seleciona apenas ativos sinteticos ativos do catalogo publico."""
    found = {}
    for row in rows:
        market = str(row.get("market", "")).lower()
        symbol = str(row.get("underlying_symbol") or row.get("symbol") or "").strip()
        if (
            market == "synthetic_index"
            and symbol
            and all(character.isalnum() or character == "_" for character in symbol)
            and row.get("is_trading_suspended") not in (1, True, "1")
        ):
            found[symbol] = str(row.get("underlying_symbol_name") or row.get("display_name") or symbol)
    return dict(sorted(found.items()))


def closed_candles(rows: list[dict[str, object]], now: int, period: int = 300) -> list[Candle]:
    """Rejeita velas em formacao, duplicadas e valores invalidos."""
    candles = {}
    for row in rows:
        try:
            epoch = int(row["epoch"])
            values = tuple(float(row[key]) for key in ("open", "high", "low", "close"))
        except (KeyError, TypeError, ValueError):
            continue
        if epoch % period or epoch + period > now or not all(isfinite(value) for value in values):
            continue
        candles[epoch] = Candle(epoch, *values)
    return [candles[epoch] for epoch in sorted(candles)]


def aggregate(candles: list[Candle], period: int) -> list[Candle]:
    """Agrupa M5 fechadas sem completar lacunas por interpolacao."""
    if period == 300:
        return list(candles)
    if period not in PERIODS:
        raise ValueError("periodo nao suportado")
    count = period // 300
    groups: dict[int, list[Candle]] = {}
    for candle in candles:
        groups.setdefault(candle.epoch // period * period, []).append(candle)
    result = []
    for epoch, group in sorted(groups.items()):
        group.sort(key=lambda candle: candle.epoch)
        if len(group) != count or any(item.epoch != epoch + index * 300 for index, item in enumerate(group)):
            continue
        result.append(
            Candle(
                epoch, group[0].open, max(item.high for item in group), min(item.low for item in group), group[-1].close
            )
        )
    return result


def latest_contiguous(candles: list[Candle], period: int) -> list[Candle]:
    """Mantem somente o bloco recente sem buracos."""
    if not candles:
        return []
    block = [candles[-1]]
    for candle in reversed(candles[:-1]):
        if candle.epoch + period != block[0].epoch:
            break
        block.insert(0, candle)
    return block

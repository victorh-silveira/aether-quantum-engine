"""Tipos e transformacoes puras do indicador de sinteticos."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite

PERIODS = (300, 900, 3600)
PERIOD_NAMES = {300: "M5", 900: "M15", 3600: "H1"}
VOLATILITY_SYMBOL = re.compile(r"(?:R_[0-9]+|1HZ[0-9]+V|RDBEAR|RDBULL)\Z")


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


@dataclass(frozen=True)
class LiveObservation:
    """Estado provisório da vela prevista, sem alterar o sinal emitido."""

    symbol: str
    period: int
    signal_epoch: int
    observed_at: int
    spot: float
    delta: float
    interim_side: str


def synthetic_symbols(rows: list[dict[str, object]], scope: str = "all") -> dict[str, str]:
    """Seleciona indices Volatility ou Stocks & Indices abertos do catalogo publico."""
    found = {}
    for row in rows:
        market = str(row.get("market", "")).lower()
        symbol = str(row.get("underlying_symbol") or row.get("symbol") or "").strip()
        suspended = row.get("is_trading_suspended") in (1, True, "1")
        if suspended:
            continue
        is_synthetic = market == "synthetic_index" and VOLATILITY_SYMBOL.fullmatch(symbol)
        is_stock_index = market == "indices" and (
            row.get("underlying_symbol_type") == "stockindex" or symbol.startswith("OTC_")
        )
        if scope == "stock_indices" and is_stock_index:
            found[symbol] = str(row.get("underlying_symbol_name") or row.get("display_name") or symbol)
        elif scope == "volatility_indices" and is_synthetic:
            found[symbol] = str(row.get("underlying_symbol_name") or row.get("display_name") or symbol)
        elif scope == "all" and (is_synthetic or is_stock_index):
            found[symbol] = str(row.get("underlying_symbol_name") or row.get("display_name") or symbol)
    return dict(sorted(found.items()))


def closed_candles(rows: list[dict[str, object]], now: int, period: int = 300) -> list[Candle]:
    """Rejeita velas em formacao, duplicadas e valores invalidos."""
    candles = {}
    conflicts = set()
    for row in rows:
        try:
            epoch = int(row["epoch"])
            values = tuple(float(row[key]) for key in ("open", "high", "low", "close"))
        except (KeyError, TypeError, ValueError):
            continue
        if epoch % period or epoch + period > now or not all(isfinite(value) for value in values):
            continue
        candle = Candle(epoch, *values)
        if epoch in candles and candles[epoch] != candle:
            conflicts.add(epoch)
        candles[epoch] = candle
    return [candles[epoch] for epoch in sorted(candles) if epoch not in conflicts]


def current_spot(rows: list[dict[str, object]], now: int) -> float | None:
    """Lê a última cotação da vela M5 em formação, sem completar lacunas."""
    expected = now // 300 * 300
    for row in reversed(rows):
        try:
            if int(row["epoch"]) != expected:
                continue
            close = float(row["close"])
        except (KeyError, TypeError, ValueError):
            continue
        if isfinite(close) and close > 0:
            return close
    return None


def live_observation(signal: Signal, spot: float, now: int) -> LiveObservation | None:
    """Compara preço atual ao fechamento usado pela previsão vigente."""
    if signal.side not in {"CALL", "PUT"} or signal.candle_epoch + signal.period > now:
        return None
    if now >= signal.candle_epoch + 2 * signal.period:
        return None
    delta = spot - signal.close
    side = "CALL" if delta > 0 else "PUT" if delta < 0 else "EMPATE"
    return LiveObservation(signal.symbol, signal.period, signal.candle_epoch, now, spot, delta, side)


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

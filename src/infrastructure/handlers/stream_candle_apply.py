"""Aplicacao incremental de velas OHLC em buffers por simbolo."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.domain.models.market_data import Candle


@dataclass(frozen=True)
class CandleApplyResult:
    """Resultado de uma atualizacao OHLC no buffer local."""

    event: str
    closed_epoch: int | None
    candle: Candle


def candle_from_ohlc(symbol: str, o: dict) -> Candle:
    """Converte payload OHLC da Deriv em Candle de dominio."""
    return Candle(
        symbol=symbol,
        open=float(o["open"]),
        high=float(o["high"]),
        low=float(o["low"]),
        close=float(o["close"]),
        time=datetime.fromtimestamp(o["open_time"]),
        epoch=int(o["open_time"]),
    )


def apply_candle_update(
    histories: dict[str, list[Candle]],
    last_epochs: dict[str, int | None],
    symbol: str,
    candle: Candle,
    *,
    limit: int,
) -> CandleApplyResult:
    """Atualiza buffer de velas e sinaliza fechamento de barra anterior."""
    history = histories.setdefault(symbol, [])
    if history and history[-1].epoch == candle.epoch:
        history[-1] = candle
        return CandleApplyResult("update", None, candle)
    prev_epoch = last_epochs.get(symbol)
    closed_epoch = None
    if prev_epoch is not None and prev_epoch != candle.epoch:
        closed_epoch = int(prev_epoch)
    history.append(candle)
    last_epochs[symbol] = candle.epoch
    if len(history) > limit:
        history.pop(0)
    event = "close" if closed_epoch is not None else "append"
    return CandleApplyResult(event, closed_epoch, candle)

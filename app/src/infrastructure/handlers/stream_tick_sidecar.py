"""Ingestao de ticks e persistencia de barras fechadas do StreamHandler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.domain.models.market_data import Candle


if TYPE_CHECKING:
    from src.infrastructure.handlers.stream_handler import StreamHandler


_TYPE_VALUE_ERRORS = (TypeError, ValueError)


async def handle_stream_tick(stream: StreamHandler, data: dict) -> None:
    """Processa ticks recebidos para microestrutura macro."""
    tick = data.get("tick")
    if not isinstance(tick, dict):
        return
    symbol = tick.get("symbol")
    if symbol not in stream.tick_buffer.symbols:
        return
    epoch = tick.get("epoch")
    quote = tick.get("quote")
    if epoch is None or quote is None:
        return
    try:
        epoch_ms = int(float(epoch) * 1000)
        price = float(quote)
    except _TYPE_VALUE_ERRORS:
        return
    stream.tick_buffer.record_tick(symbol, epoch_ms, price)
    writer = stream._market_writer
    if writer is not None:
        await writer.enqueue_tick(symbol=symbol, epoch_ms=epoch_ms, price=price)


async def persist_closed_bar(
    stream: StreamHandler,
    symbol: str,
    prev_epoch: int,
    candle: Candle,
    micro,
    *,
    granularity: int,
) -> None:
    """Envia barra fechada ao writer de series temporais."""
    writer = stream._market_writer
    if writer is None:
        return
    await writer.enqueue_bar(
        symbol=symbol,
        bar={
            "epoch": int(prev_epoch),
            "granularity": int(granularity),
            "open": float(candle.open),
            "high": float(candle.high),
            "low": float(candle.low),
            "close": float(candle.close),
            "tick_count": int(micro.tick_count) if micro is not None else 0,
            "mean_inter_tick_ms": float(micro.mean_inter_tick_ms) if micro is not None else 0.0,
            "price_velocity": float(micro.price_velocity) if micro is not None else 0.0,
        },
    )

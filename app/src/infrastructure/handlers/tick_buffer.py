"""Buffer de ticks por simbolo com agregacao de microestrutura por barra."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BarMicrostructure:
    """Stats agregadas de ticks para uma barra OHLC fechada."""

    tick_count: float
    mean_inter_tick_ms: float
    price_velocity: float
    price_acceleration: float
    consecutive_diff_std: float


_NEUTRAL = BarMicrostructure(0.0, 0.0, 0.0, 0.0, 0.0)


class TickBuffer:
    """Acumula ticks ao vivo e produz stats por barra fechada."""

    def __init__(self, symbols: list[str], *, max_ticks: int = 4096, max_bars: int = 2000):
        self._max_ticks = max(64, int(max_ticks))
        self._max_bars = max(64, int(max_bars))
        self.symbols = list(symbols)
        self._live: dict[str, deque[tuple[int, float]]] = {s: deque(maxlen=self._max_ticks) for s in symbols}
        self._bar_stats: dict[str, deque[BarMicrostructure]] = {s: deque(maxlen=self._max_bars) for s in symbols}
        self._current_epoch: dict[str, int | None] = dict.fromkeys(symbols)
        self._last_tick_monotonic: float = 0.0

    def touch_activity(self) -> None:
        """Marca atividade de ingestao no relogio monotonico do loop asyncio."""
        try:
            self._last_tick_monotonic = asyncio.get_running_loop().time()
        except RuntimeError:
            self._last_tick_monotonic = 0.0

    def last_tick_monotonic(self) -> float:
        """Retorna timestamp monotonico do ultimo tick registrado."""
        return float(self._last_tick_monotonic)

    def record_tick(self, symbol: str, epoch_ms: int, price: float) -> None:
        """Registra um tick recebido do WebSocket."""
        if symbol not in self._live:
            return
        self._live[symbol].append((int(epoch_ms), float(price)))
        self.touch_activity()

    def on_bar_close(self, symbol: str, bar_epoch: int) -> BarMicrostructure:
        """Finaliza stats da barra e reinicia acumulador para a proxima."""
        stats = self._aggregate_bar(symbol)
        if symbol in self._bar_stats:
            self._bar_stats[symbol].append(stats)
        self._current_epoch[symbol] = int(bar_epoch)
        self._live[symbol].clear()
        return stats

    def on_bar_update(self, symbol: str, bar_epoch: int) -> None:
        """Marca epoch da barra em formacao."""
        self._current_epoch[symbol] = int(bar_epoch)

    def microstructure_series(self, symbol: str, length: int) -> list[BarMicrostructure]:
        """Retorna serie alinhada ao fim do historico de velas (padding neutro a esquerda)."""
        hist = list(self._bar_stats.get(symbol, []))
        if length <= 0:
            return []
        if len(hist) >= length:
            return hist[-length:]
        pad = [_NEUTRAL] * (length - len(hist))
        return pad + hist

    def microstructure_arrays(self, symbol: str, length: int) -> dict[str, np.ndarray]:
        """Converte serie de microestrutura em arrays numpy por campo."""
        series = self.microstructure_series(symbol, length)
        return {
            "tick_count": np.array([s.tick_count for s in series], dtype=np.float64),
            "mean_inter_tick_ms": np.array([s.mean_inter_tick_ms for s in series], dtype=np.float64),
            "price_velocity": np.array([s.price_velocity for s in series], dtype=np.float64),
            "price_acceleration": np.array([s.price_acceleration for s in series], dtype=np.float64),
            "consecutive_diff_std": np.array([s.consecutive_diff_std for s in series], dtype=np.float64),
        }

    def _aggregate_bar(self, symbol: str) -> BarMicrostructure:
        """Calcula microestrutura a partir dos ticks acumulados na barra."""
        ticks = list(self._live.get(symbol, []))
        if len(ticks) < 2:
            return _NEUTRAL
        epochs = np.array([t[0] for t in ticks], dtype=np.float64)
        prices = np.array([t[1] for t in ticks], dtype=np.float64)
        diffs = np.diff(epochs)
        valid = diffs > 0
        if not np.any(valid):
            return BarMicrostructure(float(len(ticks)), 0.0, 0.0, 0.0, 0.0)
        mean_ms = float(np.mean(diffs[valid]))
        price_diffs = np.diff(prices)
        velocity = float(np.mean(price_diffs / np.maximum(diffs, 1.0)))
        if len(price_diffs) >= 2 and len(diffs) >= 2:
            v1 = price_diffs[:-1] / np.maximum(diffs[:-1], 1.0)
            v2 = price_diffs[1:] / np.maximum(diffs[1:], 1.0)
            acceleration = float(np.mean(v2 - v1))
        else:
            acceleration = 0.0
        diff_std = float(np.std(price_diffs)) if len(price_diffs) > 0 else 0.0
        return BarMicrostructure(
            tick_count=float(len(ticks)),
            mean_inter_tick_ms=mean_ms,
            price_velocity=velocity,
            price_acceleration=acceleration,
            consecutive_diff_std=diff_std,
        )

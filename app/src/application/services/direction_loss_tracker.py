"""Rastreador de perdas consecutivas por direcao para anti-trend-lock."""

from __future__ import annotations

import asyncio
import time

from src.domain.models.trade import TradeDirection


_DIRECTION_KEYS = ("CALL", "PUT")


class _DirectionLossTrackerHolder:
    """Armazena instancia singleton do rastreador de perdas direcionais."""

    tracker: DirectionLossTracker | None = None


def _loss_timestamp_key(symbol: str, direction: str) -> str:
    """Monta chave canonica para timestamp de perda direcional."""
    return f"{symbol}:{direction}"


def _cooperative_loop_time() -> float:
    """Retorna tempo monotono do loop asyncio ativo ou fallback sincrono."""
    try:
        return asyncio.get_running_loop().time()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                return loop.time()
        except RuntimeError:
            pass
        return time.monotonic()


class DirectionLossTracker:
    """Memoria de perdas direcionais com expiracao temporal cooperativa."""

    def __init__(self) -> None:
        self._loss_tracker: dict[str, dict[str, int]] = {}
        self._last_loss_timestamp: dict[str, float] = {}

    def reset(self) -> None:
        """Limpa contadores e timestamps de perdas direcionais."""
        self._loss_tracker.clear()
        self._last_loss_timestamp.clear()

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Retorna copia imutavel do rastreador de perdas por direcao."""
        return {symbol: dict(counts) for symbol, counts in self._loss_tracker.items()}

    def consecutive_losses(self, symbol: str, direction: str) -> int:
        """Retorna perdas consecutivas na mesma direcao para o simbolo."""
        sym = str(symbol)
        dir_key = str(direction or "").upper()
        if dir_key not in _DIRECTION_KEYS:
            return 0
        return int(self._loss_tracker.get(sym, {}).get(dir_key, 0))

    def record_outcome(self, symbol: str, direction: str | None, *, won: bool) -> None:
        """Atualiza perdas consecutivas por direcao apos liquidacao."""
        sym = str(symbol)
        dir_key = str(direction or "").upper()
        if dir_key not in _DIRECTION_KEYS:
            return
        bucket = self._loss_tracker.setdefault(sym, dict.fromkeys(_DIRECTION_KEYS, 0))
        ts_key = _loss_timestamp_key(sym, dir_key)
        if won:
            bucket[dir_key] = 0
            self._last_loss_timestamp.pop(ts_key, None)
            return
        bucket[dir_key] = int(bucket.get(dir_key, 0)) + 1
        self._last_loss_timestamp[ts_key] = _cooperative_loop_time()

    def anti_trend_lock_active(self, symbol: str, direction: TradeDirection) -> bool:
        """Indica bloqueio anti-trend-lock apos o limiar de duas perdas consecutivas."""
        return self.consecutive_losses(symbol, direction.name) >= 2

    def prune_obsolete_direction_losses(self, max_age_seconds: float = 120.0) -> None:
        """Expira memoria de stress obsoleta quando a ultima perda excede o TTL."""
        now = _cooperative_loop_time()
        for sym, bucket in list(self._loss_tracker.items()):
            for dir_key in _DIRECTION_KEYS:
                if int(bucket.get(dir_key, 0)) <= 0:
                    continue
                ts_key = _loss_timestamp_key(sym, dir_key)
                last_ts = self._last_loss_timestamp.get(ts_key)
                if last_ts is None:
                    continue
                if (now - last_ts) > float(max_age_seconds):
                    bucket[dir_key] = 0
                    self._last_loss_timestamp.pop(ts_key, None)


def get_direction_loss_tracker() -> DirectionLossTracker:
    """Retorna singleton do rastreador de perdas direcionais."""
    if _DirectionLossTrackerHolder.tracker is None:
        _DirectionLossTrackerHolder.tracker = DirectionLossTracker()
    return _DirectionLossTrackerHolder.tracker


def reset_direction_persistence_tracker() -> None:
    """Limpa rastreador de perdas sequenciais por direcao."""
    get_direction_loss_tracker().reset()


def direction_loss_tracker_snapshot() -> dict[str, dict[str, int]]:
    """Retorna copia imutavel do rastreador de perdas por direcao."""
    return get_direction_loss_tracker().snapshot()


def consecutive_direction_losses(symbol: str, direction: str) -> int:
    """Retorna perdas consecutivas na mesma direcao para o simbolo."""
    return get_direction_loss_tracker().consecutive_losses(symbol, direction)


def record_direction_outcome(symbol: str, direction: str | None, *, won: bool) -> None:
    """Atualiza perdas consecutivas por direcao apos liquidacao."""
    get_direction_loss_tracker().record_outcome(symbol, direction, won=won)


def anti_trend_lock_active(symbol: str, direction: TradeDirection) -> bool:
    """Indica bloqueio anti-trend-lock apos o limiar de duas perdas consecutivas."""
    return get_direction_loss_tracker().anti_trend_lock_active(symbol, direction)

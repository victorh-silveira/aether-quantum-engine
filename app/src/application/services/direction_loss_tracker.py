"""Rastreador de perdas consecutivas por direcao para anti-trend-lock."""

from __future__ import annotations

from src.domain.models.trade import TradeDirection


_DIRECTION_KEYS = ("CALL", "PUT")
_loss_tracker: dict[str, dict[str, int]] = {}


def reset_direction_persistence_tracker() -> None:
    """Limpa rastreador de perdas sequenciais por direcao."""
    _loss_tracker.clear()


def direction_loss_tracker_snapshot() -> dict[str, dict[str, int]]:
    """Retorna copia imutavel do rastreador de perdas por direcao."""
    return {symbol: dict(counts) for symbol, counts in _loss_tracker.items()}


def consecutive_direction_losses(symbol: str, direction: str) -> int:
    """Retorna perdas consecutivas na mesma direcao para o simbolo."""
    sym = str(symbol)
    dir_key = str(direction or "").upper()
    if dir_key not in _DIRECTION_KEYS:
        return 0
    return int(_loss_tracker.get(sym, {}).get(dir_key, 0))


def record_direction_outcome(symbol: str, direction: str | None, *, won: bool) -> None:
    """Atualiza perdas consecutivas por direcao apos liquidacao."""
    sym = str(symbol)
    dir_key = str(direction or "").upper()
    if dir_key not in _DIRECTION_KEYS:
        return
    bucket = _loss_tracker.setdefault(sym, dict.fromkeys(_DIRECTION_KEYS, 0))
    if won:
        bucket[dir_key] = 0
        return
    bucket[dir_key] = int(bucket.get(dir_key, 0)) + 1


def anti_trend_lock_active(symbol: str, direction: TradeDirection) -> bool:
    """Indica bloqueio anti-trend-lock apos o limiar de duas perdas consecutivas."""
    return consecutive_direction_losses(symbol, direction.name) >= 2

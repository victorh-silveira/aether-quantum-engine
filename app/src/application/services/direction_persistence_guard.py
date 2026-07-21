"""Filtro de persistencia direcional anti-trend-lock com freeze cross-symbol."""

from __future__ import annotations

from typing import Any

from src.application.services.direction_loss_tracker import consecutive_direction_losses
from src.application.services.direction_persistence_guard_part2 import (
    _freeze_cycle,
    _mark_guard,
    _resolve_peer_lock,
    log_regime_guard,
)
from src.application.services.execution_runtime_config import resolve_direction_persistence_config
from src.application.services.regime_micro_freeze import apply_regime_freeze_if_congested
from src.domain.models.trade import TradeDirection


__all__ = ["evaluate_direction_persistence_guard", "log_regime_guard"]


def _guard_blocked_same_direction(
    symbol: str, proposed: TradeDirection, metrics: dict[str, Any], cycle_id: int
) -> TradeDirection | None:
    """Bloqueia repeticao da mesma direcao apos duas perdas consecutivas."""
    count = consecutive_direction_losses(symbol, proposed.name)
    if count >= int(resolve_direction_persistence_config()["same_direction_count_threshold"]):
        _mark_guard(metrics, count)
        if apply_regime_freeze_if_congested(metrics, persistence_filter_active=True):
            _freeze_cycle(metrics, cycle_id, count)
        return None
    return proposed


def evaluate_direction_persistence_guard(
    symbol: str | None,
    dl_dir: TradeDirection,
    proposed: TradeDirection,
    metrics: dict[str, Any],
    *,
    entry: dict[str, Any],
    peer_entry: dict[str, Any] | None,
    cycle_id: int,
    infra_cfg: dict | None,
) -> TradeDirection | None:
    """Aplica anti-trend-lock com freeze cross-symbol ou congelamento por congestao."""
    if not symbol:
        return proposed
    blocked = _guard_blocked_same_direction(symbol, proposed, metrics, cycle_id)
    if blocked is None:
        return None
    if isinstance(peer_entry, dict):
        locked = _resolve_peer_lock(
            symbol, dl_dir, metrics, entry=entry, peer_entry=peer_entry, cycle_id=cycle_id, infra_cfg=infra_cfg
        )
        if locked is not None or metrics.get("signal_status") == "SIGNAL_SUSPENDED":
            return locked
    return proposed

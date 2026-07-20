"""Parte 2 do filtro de persistencia direcional: FREEZE/SKIP sem flip de lado."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.direction_loss_tracker import anti_trend_lock_active, consecutive_direction_losses
from src.application.services.direction_persistence_guard_helpers import (
    _LOGGED_REGIME_GUARD_CYCLES,
    prune_regime_guard_log_state,
)
from src.application.services.meta_classifier_cross_symbol import ANCHOR_BEAR, ANCHOR_BULL
from src.application.services.regime_micro_freeze import apply_regime_freeze_if_congested
from src.domain.models.trade import TradeDirection


_GUARD_LOGGER = logging.getLogger("AETH")


def log_regime_guard(cycle_id: int, action: str, consecutive_losses: int) -> None:
    """Emite telemetria padronizada do filtro anti-trend-lock."""
    cid = int(cycle_id)
    action_text = str(action)
    if action_text == "FREEZE: SKIP CYCLE":
        seen = _LOGGED_REGIME_GUARD_CYCLES.get(cid, frozenset())
        if action_text in seen:
            return
        _LOGGED_REGIME_GUARD_CYCLES[cid] = seen | frozenset({action_text})
        prune_regime_guard_log_state(cid)
    _GUARD_LOGGER.info(
        "[C%04d] REGIME_GUARD | {AntiTrendLock: %s} | consecutive_losses=%d", cid, action_text, int(consecutive_losses)
    )


def _mark_guard(metrics: dict[str, Any], count: int) -> None:
    """Marca metricas quando o filtro anti-trend-lock esta ativo."""
    metrics["anti_trend_lock_active"] = True
    metrics["consecutive_direction_losses"] = int(count)


def _freeze_cycle(metrics: dict[str, Any], cycle_id: int, count: int) -> None:
    """Suspende o ciclo corrente por congestao micro CHOP_CONGESTION."""
    metrics["signal_status"] = "SIGNAL_SUSPENDED"
    metrics["regime_classification"] = "CHOP_CONGESTION"
    metrics["regime_guard_action"] = "FREEZE: SKIP CYCLE"
    log_regime_guard(cycle_id, "FREEZE: SKIP CYCLE", count)


def _freeze_peer_lock(metrics: dict[str, Any], *, lock_symbol: str, lock_direction: str, cycle_id: int) -> None:
    """Congela ciclo quando lock peer esta ativo; nunca inverte CALL/PUT."""
    count = consecutive_direction_losses(lock_symbol, lock_direction)
    _mark_guard(metrics, count)
    if apply_regime_freeze_if_congested(metrics, persistence_filter_active=True):
        _freeze_cycle(metrics, cycle_id, count)
        return
    _freeze_cycle(metrics, cycle_id, count)


def _attempt_bull_call_lock_freeze(
    dl_dir: TradeDirection,
    metrics: dict[str, Any],
    *,
    entry: dict[str, Any],
    peer_entry: dict[str, Any],
    cycle_id: int,
    infra_cfg: dict | None,
) -> TradeDirection | None:
    """Congela ciclo sob lock CALL no ancla bull; sem flip de direcao."""
    _ = (dl_dir, entry, peer_entry, infra_cfg)
    _freeze_peer_lock(metrics, lock_symbol=ANCHOR_BULL, lock_direction="CALL", cycle_id=cycle_id)
    return None


def _attempt_bull_put_lock_freeze(
    dl_dir: TradeDirection,
    metrics: dict[str, Any],
    *,
    entry: dict[str, Any],
    peer_entry: dict[str, Any],
    cycle_id: int,
    infra_cfg: dict | None,
) -> TradeDirection | None:
    """Congela ciclo sob lock PUT no ancla bull; sem flip de direcao."""
    _ = (dl_dir, entry, peer_entry, infra_cfg)
    _freeze_peer_lock(metrics, lock_symbol=ANCHOR_BULL, lock_direction="PUT", cycle_id=cycle_id)
    return None


def _attempt_bear_put_lock_freeze(
    dl_dir: TradeDirection,
    metrics: dict[str, Any],
    *,
    entry: dict[str, Any],
    peer_entry: dict[str, Any],
    cycle_id: int,
    infra_cfg: dict | None,
) -> TradeDirection | None:
    """Congela ciclo sob lock PUT no ancla bear; sem flip de direcao."""
    _ = (dl_dir, entry, peer_entry, infra_cfg)
    _freeze_peer_lock(metrics, lock_symbol=ANCHOR_BEAR, lock_direction="PUT", cycle_id=cycle_id)
    return None


def _attempt_bear_call_lock_freeze(
    dl_dir: TradeDirection,
    metrics: dict[str, Any],
    *,
    entry: dict[str, Any],
    peer_entry: dict[str, Any],
    cycle_id: int,
    infra_cfg: dict | None,
) -> TradeDirection | None:
    """Congela ciclo sob lock CALL no ancla bear; sem flip de direcao."""
    _ = (dl_dir, entry, peer_entry, infra_cfg)
    _freeze_peer_lock(metrics, lock_symbol=ANCHOR_BEAR, lock_direction="CALL", cycle_id=cycle_id)
    return None


def _resolve_peer_lock(
    symbol: str,
    dl_dir: TradeDirection,
    metrics: dict[str, Any],
    *,
    entry: dict[str, Any],
    peer_entry: dict[str, Any],
    cycle_id: int,
    infra_cfg: dict | None,
) -> TradeDirection | None:
    """Peer lock so congela ciclo; nunca retorna direcao invertida."""
    if ANCHOR_BULL == ANCHOR_BEAR:
        return None
    if symbol == ANCHOR_BEAR and anti_trend_lock_active(ANCHOR_BULL, TradeDirection.CALL):
        return _attempt_bull_call_lock_freeze(
            dl_dir, metrics, entry=entry, peer_entry=peer_entry, cycle_id=cycle_id, infra_cfg=infra_cfg
        )
    if symbol == ANCHOR_BEAR and anti_trend_lock_active(ANCHOR_BULL, TradeDirection.PUT):
        return _attempt_bull_put_lock_freeze(
            dl_dir, metrics, entry=entry, peer_entry=peer_entry, cycle_id=cycle_id, infra_cfg=infra_cfg
        )
    if symbol == ANCHOR_BULL and anti_trend_lock_active(ANCHOR_BEAR, TradeDirection.PUT):
        return _attempt_bear_put_lock_freeze(
            dl_dir, metrics, entry=entry, peer_entry=peer_entry, cycle_id=cycle_id, infra_cfg=infra_cfg
        )
    if symbol == ANCHOR_BULL and anti_trend_lock_active(ANCHOR_BEAR, TradeDirection.CALL):
        return _attempt_bear_call_lock_freeze(
            dl_dir, metrics, entry=entry, peer_entry=peer_entry, cycle_id=cycle_id, infra_cfg=infra_cfg
        )
    return None

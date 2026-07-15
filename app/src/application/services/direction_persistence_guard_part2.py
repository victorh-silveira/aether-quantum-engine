"""Parte 2 do filtro de persistencia direcional anti-trend-lock com flip cross-symbol."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.direction_loss_tracker import (
    anti_trend_lock_active,
    consecutive_direction_losses,
)
from src.application.services.direction_persistence_guard_helpers import (
    _LOGGED_REGIME_GUARD_CYCLES,
    cross_prob_delta_mean,
    entry_prob,
    peer_payoff_edge,
    prune_regime_guard_log_state,
)
from src.application.services.execution_quality_gate import sync_direction_margin
from src.application.services.meta_classifier_cross_symbol import ANCHOR_BEAR, ANCHOR_BULL
from src.application.services.meta_classifier_features import cross_symbol_conviction_spread
from src.application.services.meta_direction_flip import (
    META_FLIP_TRADE_SCORE,
    apply_regime_freeze_if_congested,
)
from src.domain.models.trade import TradeDirection
from src.domain.risk.risk_recovery_state import evaluate_anti_trend_lock


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
        "[C%04d] REGIME_GUARD | {AntiTrendLock: %s} | consecutive_losses=%d",
        cid,
        action_text,
        int(consecutive_losses),
    )


def _apply_flip_metrics(
    metrics: dict[str, Any],
    *,
    dl_dir: TradeDirection,
    exec_dir: TradeDirection,
    action: str,
) -> None:
    """Propaga telemetria de flip anti-trend-lock nas metricas do ciclo."""
    score = float(META_FLIP_TRADE_SCORE)
    metrics["anti_trend_lock_flip"] = True
    metrics["meta_direction_flip"] = True
    metrics["direction_inverted"] = dl_dir != exec_dir
    metrics["regime_guard_action"] = action
    metrics["dl_direction"] = dl_dir.name
    metrics["exec_direction"] = exec_dir.name
    metrics["resolved_direction"] = exec_dir.name
    metrics["trade_score"] = score
    metrics["conviction"] = score
    if exec_dir == TradeDirection.CALL:
        metrics["direction_call_score"] = score
        metrics["direction_put_score"] = max(0.0, 1.0 - score)
    else:
        metrics["direction_put_score"] = score
        metrics["direction_call_score"] = max(0.0, 1.0 - score)
    sync_direction_margin(metrics, direction=exec_dir.name)


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


def _attempt_bull_call_lock_flip(
    dl_dir: TradeDirection,
    metrics: dict[str, Any],
    *,
    entry: dict[str, Any],
    peer_entry: dict[str, Any],
    cycle_id: int,
    infra_cfg: dict | None,
) -> TradeDirection | None:
    """Forca flip para PUT no RDBEAR apos sequencia de CALL perdidos no RDBULL."""
    count = consecutive_direction_losses(ANCHOR_BULL, "CALL")
    _mark_guard(metrics, count)
    if apply_regime_freeze_if_congested(metrics, persistence_filter_active=True):
        _freeze_cycle(metrics, cycle_id, count)
        return None

    bull_call_prob, bear_put_prob = entry_prob(peer_entry), 1.0 - entry_prob(entry)
    predicted_payoff_edge = peer_payoff_edge(entry, metrics)
    cross_symbol_prob_delta_mean = cross_prob_delta_mean(metrics, infra_cfg)
    probability_delta = cross_symbol_conviction_spread(metrics)

    resolved_dir, action = evaluate_anti_trend_lock(
        ANCHOR_BULL,
        TradeDirection.CALL,
        count,
        bull_call_prob,
        bear_put_prob,
        probability_delta,
        predicted_payoff_edge,
        cross_symbol_prob_delta_mean,
    )

    if resolved_dir == TradeDirection.PUT and action == "FLIP to PUT":
        log_regime_guard(cycle_id, "FLIP to PUT", count)
        _apply_flip_metrics(metrics, dl_dir=dl_dir, exec_dir=TradeDirection.PUT, action=action)
        return TradeDirection.PUT

    _freeze_cycle(metrics, cycle_id, count)
    return None


def _attempt_bull_put_lock_flip(
    dl_dir: TradeDirection,
    metrics: dict[str, Any],
    *,
    entry: dict[str, Any],
    peer_entry: dict[str, Any],
    cycle_id: int,
    infra_cfg: dict | None,
) -> TradeDirection | None:
    """Forca flip para CALL no RDBEAR apos sequencia de PUT perdidos no RDBULL."""
    count = consecutive_direction_losses(ANCHOR_BULL, "PUT")
    _mark_guard(metrics, count)
    if apply_regime_freeze_if_congested(metrics, persistence_filter_active=True):
        _freeze_cycle(metrics, cycle_id, count)
        return None

    bull_call_prob, bear_put_prob = entry_prob(peer_entry), 1.0 - entry_prob(entry)
    predicted_payoff_edge = peer_payoff_edge(entry, metrics)
    cross_symbol_prob_delta_mean = cross_prob_delta_mean(metrics, infra_cfg)
    probability_delta = cross_symbol_conviction_spread(metrics)

    resolved_dir, action = evaluate_anti_trend_lock(
        ANCHOR_BULL,
        TradeDirection.PUT,
        count,
        bull_call_prob,
        bear_put_prob,
        probability_delta,
        predicted_payoff_edge,
        cross_symbol_prob_delta_mean,
    )

    if resolved_dir == TradeDirection.CALL and action == "FLIP to CALL":
        log_regime_guard(cycle_id, "FLIP to CALL", count)
        _apply_flip_metrics(metrics, dl_dir=dl_dir, exec_dir=TradeDirection.CALL, action=action)
        return TradeDirection.CALL

    _freeze_cycle(metrics, cycle_id, count)
    return None


def _attempt_bear_put_lock_flip(
    dl_dir: TradeDirection,
    metrics: dict[str, Any],
    *,
    entry: dict[str, Any],
    peer_entry: dict[str, Any],
    cycle_id: int,
    infra_cfg: dict | None,
) -> TradeDirection | None:
    """Forca flip para CALL no RDBULL apos sequencia de PUT perdidos no RDBEAR."""
    count = consecutive_direction_losses(ANCHOR_BEAR, "PUT")
    _mark_guard(metrics, count)
    if apply_regime_freeze_if_congested(metrics, persistence_filter_active=True):
        _freeze_cycle(metrics, cycle_id, count)
        return None

    bull_call_prob, bear_put_prob = entry_prob(entry), 1.0 - entry_prob(peer_entry)
    predicted_payoff_edge = peer_payoff_edge(entry, metrics)
    cross_symbol_prob_delta_mean = cross_prob_delta_mean(metrics, infra_cfg)
    probability_delta = cross_symbol_conviction_spread(metrics)

    resolved_dir, action = evaluate_anti_trend_lock(
        ANCHOR_BEAR,
        TradeDirection.PUT,
        count,
        bull_call_prob,
        bear_put_prob,
        probability_delta,
        predicted_payoff_edge,
        cross_symbol_prob_delta_mean,
    )

    if resolved_dir == TradeDirection.CALL and action == "FLIP to CALL":
        log_regime_guard(cycle_id, "FLIP to CALL", count)
        _apply_flip_metrics(metrics, dl_dir=dl_dir, exec_dir=TradeDirection.CALL, action=action)
        return TradeDirection.CALL

    _freeze_cycle(metrics, cycle_id, count)
    return None


def _attempt_bear_call_lock_flip(
    dl_dir: TradeDirection,
    metrics: dict[str, Any],
    *,
    entry: dict[str, Any],
    peer_entry: dict[str, Any],
    cycle_id: int,
    infra_cfg: dict | None,
) -> TradeDirection | None:
    """Forca flip para PUT no RDBULL apos sequencia de CALL perdidos no RDBEAR."""
    count = consecutive_direction_losses(ANCHOR_BEAR, "CALL")
    _mark_guard(metrics, count)
    if apply_regime_freeze_if_congested(metrics, persistence_filter_active=True):
        _freeze_cycle(metrics, cycle_id, count)
        return None

    bull_call_prob, bear_put_prob = entry_prob(entry), 1.0 - entry_prob(peer_entry)
    predicted_payoff_edge = peer_payoff_edge(entry, metrics)
    cross_symbol_prob_delta_mean = cross_prob_delta_mean(metrics, infra_cfg)
    probability_delta = cross_symbol_conviction_spread(metrics)

    resolved_dir, action = evaluate_anti_trend_lock(
        ANCHOR_BEAR,
        TradeDirection.CALL,
        count,
        bull_call_prob,
        bear_put_prob,
        probability_delta,
        predicted_payoff_edge,
        cross_symbol_prob_delta_mean,
    )

    if resolved_dir == TradeDirection.PUT and action == "FLIP to PUT":
        log_regime_guard(cycle_id, "FLIP to PUT", count)
        _apply_flip_metrics(metrics, dl_dir=dl_dir, exec_dir=TradeDirection.PUT, action=action)
        return TradeDirection.PUT

    _freeze_cycle(metrics, cycle_id, count)
    return None


def _resolve_peer_flip(
    symbol: str,
    dl_dir: TradeDirection,
    metrics: dict[str, Any],
    *,
    entry: dict[str, Any],
    peer_entry: dict[str, Any],
    cycle_id: int,
    infra_cfg: dict | None,
) -> TradeDirection | None:
    """Delega flip cross-symbol conforme lock ativo no par Drift."""
    if symbol == ANCHOR_BEAR and anti_trend_lock_active(ANCHOR_BULL, TradeDirection.CALL):
        return _attempt_bull_call_lock_flip(
            dl_dir, metrics, entry=entry, peer_entry=peer_entry, cycle_id=cycle_id, infra_cfg=infra_cfg
        )
    if symbol == ANCHOR_BEAR and anti_trend_lock_active(ANCHOR_BULL, TradeDirection.PUT):
        return _attempt_bull_put_lock_flip(
            dl_dir, metrics, entry=entry, peer_entry=peer_entry, cycle_id=cycle_id, infra_cfg=infra_cfg
        )
    if symbol == ANCHOR_BULL and anti_trend_lock_active(ANCHOR_BEAR, TradeDirection.PUT):
        return _attempt_bear_put_lock_flip(
            dl_dir, metrics, entry=entry, peer_entry=peer_entry, cycle_id=cycle_id, infra_cfg=infra_cfg
        )
    if symbol == ANCHOR_BULL and anti_trend_lock_active(ANCHOR_BEAR, TradeDirection.CALL):
        return _attempt_bear_call_lock_flip(
            dl_dir, metrics, entry=entry, peer_entry=peer_entry, cycle_id=cycle_id, infra_cfg=infra_cfg
        )
    return None

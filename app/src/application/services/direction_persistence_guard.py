"""Filtro de persistencia direcional anti-trend-lock com flip cross-symbol."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.direction_loss_tracker import (
    anti_trend_lock_active,
    consecutive_direction_losses,
)
from src.application.services.execution_quality_gate import sync_direction_margin
from src.application.services.meta_classifier_cross_symbol import ANCHOR_BEAR, ANCHOR_BULL
from src.application.services.meta_classifier_features import cross_symbol_conviction_spread
from src.application.services.meta_direction_flip import (
    META_FLIP_TRADE_SCORE,
    apply_regime_freeze_if_congested,
)
from src.domain.models.trade import TradeDirection


_GUARD_LOGGER = logging.getLogger("AETH")
_LOGGED_REGIME_GUARD_CYCLES: dict[int, frozenset[str]] = {}


def reset_regime_guard_log_state() -> None:
    """Limpa deduplicacao de logs do guard para testes e reinicios de sessao."""
    _LOGGED_REGIME_GUARD_CYCLES.clear()


def _prune_regime_guard_log_state(current_cycle_id: int) -> None:
    """Remove entradas antigas da deduplicacao de logs do guard."""
    stale = [key for key in _LOGGED_REGIME_GUARD_CYCLES if key < int(current_cycle_id) - 100]
    for key in stale:
        _LOGGED_REGIME_GUARD_CYCLES.pop(key, None)


def _entry_prob(entry: dict[str, Any]) -> float:
    """Retorna probabilidade calibrada limitada do entry de decisao."""
    metrics = entry.get("metrics") or {}
    raw = metrics.get("calibrated_prob", metrics.get("raw_prob"))
    if raw is None:
        return 0.5
    return max(0.0, min(1.0, float(raw)))


def _cross_prob_delta_mean(metrics: dict[str, Any], infra_cfg: dict | None) -> float:
    """Retorna media historica do spread cross-symbol para comparacao de expansao."""
    stored = metrics.get("cross_symbol_prob_delta_mean")
    if stored is not None:
        return float(stored)
    infra = (infra_cfg or {}).get("meta_classifier") or {}
    manifest = infra.get("cross_symbol_prob_delta_mean")
    if manifest is not None:
        return float(manifest)
    return 0.0


def _peer_payoff_edge(peer_entry: dict[str, Any] | None, metrics: dict[str, Any]) -> float:
    """Le predicted_payoff_edge do par alternativo com fallback local."""
    if isinstance(peer_entry, dict):
        peer_metrics = peer_entry.get("metrics") or {}
        if peer_metrics.get("predicted_payoff_edge") is not None:
            return float(peer_metrics["predicted_payoff_edge"])
    if metrics.get("predicted_payoff_edge") is not None:
        return float(metrics["predicted_payoff_edge"])
    return 0.0


def bear_put_prob_expanding(
    bull_entry: dict[str, Any],
    bear_entry: dict[str, Any] | None,
    metrics: dict[str, Any],
    infra_cfg: dict | None,
) -> bool:
    """Indica expansao de probabilidade PUT no RDBEAR versus CALL no RDBULL."""
    if not isinstance(bear_entry, dict):
        return False
    bull_call = _entry_prob(bull_entry)
    bear_put = 1.0 - _entry_prob(bear_entry)
    metrics["bear_prob_put"] = float(bear_put)
    delta = cross_symbol_conviction_spread(metrics)
    mean = _cross_prob_delta_mean(metrics, infra_cfg)
    return bear_put + 1e-12 > bull_call or delta > mean + 1e-12


def bull_call_prob_expanding(
    bear_entry: dict[str, Any],
    bull_entry: dict[str, Any] | None,
    metrics: dict[str, Any],
    infra_cfg: dict | None,
) -> bool:
    """Indica expansao de probabilidade CALL no RDBULL versus PUT no RDBEAR."""
    if not isinstance(bull_entry, dict):
        return False
    bear_put = 1.0 - _entry_prob(bear_entry)
    bull_call = _entry_prob(bull_entry)
    metrics["bull_prob_call"] = float(bull_call)
    delta = cross_symbol_conviction_spread(metrics)
    mean = _cross_prob_delta_mean(metrics, infra_cfg)
    return bull_call + 1e-12 > bear_put or delta > mean + 1e-12


def log_regime_guard(cycle_id: int, action: str, consecutive_losses: int) -> None:
    """Emite telemetria padronizada do filtro anti-trend-lock."""
    cid = int(cycle_id)
    action_text = str(action)
    if action_text == "FREEZE: SKIP CYCLE":
        seen = _LOGGED_REGIME_GUARD_CYCLES.get(cid, frozenset())
        if action_text in seen:
            return
        _LOGGED_REGIME_GUARD_CYCLES[cid] = seen | frozenset({action_text})
        _prune_regime_guard_log_state(cid)
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


def _guard_blocked_same_direction(
    symbol: str,
    proposed: TradeDirection,
    metrics: dict[str, Any],
    cycle_id: int,
) -> TradeDirection | None:
    """Bloqueia repeticao da mesma direcao apos duas perdas consecutivas."""
    if not anti_trend_lock_active(symbol, proposed):
        return proposed
    count = consecutive_direction_losses(symbol, proposed.name)
    _mark_guard(metrics, count)
    if apply_regime_freeze_if_congested(metrics, persistence_filter_active=True):
        _freeze_cycle(metrics, cycle_id, count)
        return None
    if symbol == ANCHOR_BULL and proposed == TradeDirection.CALL:
        return None
    if symbol == ANCHOR_BEAR and proposed == TradeDirection.PUT:
        return None
    return None


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
    outcome: TradeDirection | None = None
    if bear_put_prob_expanding(peer_entry, entry, metrics, infra_cfg):
        edge = _peer_payoff_edge(entry, metrics)
        if edge + 1e-12 >= 0.0:
            log_regime_guard(cycle_id, "FLIP to PUT", count)
            _apply_flip_metrics(metrics, dl_dir=dl_dir, exec_dir=TradeDirection.PUT, action="FLIP to PUT")
            outcome = TradeDirection.PUT
    if outcome is None:
        _freeze_cycle(metrics, cycle_id, count)
    return outcome


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
    outcome: TradeDirection | None = None
    if bull_call_prob_expanding(peer_entry, entry, metrics, infra_cfg):
        edge = _peer_payoff_edge(entry, metrics)
        if edge + 1e-12 >= 0.0:
            log_regime_guard(cycle_id, "FLIP to CALL", count)
            _apply_flip_metrics(metrics, dl_dir=dl_dir, exec_dir=TradeDirection.CALL, action="FLIP to CALL")
            outcome = TradeDirection.CALL
    if outcome is None:
        _freeze_cycle(metrics, cycle_id, count)
    return outcome


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
            dl_dir,
            metrics,
            entry=entry,
            peer_entry=peer_entry,
            cycle_id=cycle_id,
            infra_cfg=infra_cfg,
        )
    if symbol == ANCHOR_BULL and anti_trend_lock_active(ANCHOR_BEAR, TradeDirection.PUT):
        return _attempt_bear_put_lock_flip(
            dl_dir,
            metrics,
            entry=entry,
            peer_entry=peer_entry,
            cycle_id=cycle_id,
            infra_cfg=infra_cfg,
        )
    return None


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
    """Aplica anti-trend-lock com flip cross-symbol ou congelamento por congestao."""
    if not symbol:
        return proposed
    blocked = _guard_blocked_same_direction(symbol, proposed, metrics, cycle_id)
    if blocked is None:
        return None
    if isinstance(peer_entry, dict):
        flipped = _resolve_peer_flip(
            symbol,
            dl_dir,
            metrics,
            entry=entry,
            peer_entry=peer_entry,
            cycle_id=cycle_id,
            infra_cfg=infra_cfg,
        )
        if flipped is not None or metrics.get("signal_status") == "SIGNAL_SUSPENDED":
            return flipped
    return proposed

"""Auxiliares e utilitários para o filtro de persistência anti-trend-lock."""

from __future__ import annotations

from typing import Any

from src.application.services.meta_classifier_features import cross_symbol_conviction_spread


_LOGGED_REGIME_GUARD_CYCLES: dict[int, frozenset[str]] = {}


def reset_regime_guard_log_state() -> None:
    """Limpa deduplicacao de logs do guard para testes e reinicios de sessao."""
    _LOGGED_REGIME_GUARD_CYCLES.clear()


def prune_regime_guard_log_state(current_cycle_id: int) -> None:
    """Remove entradas antigas da deduplicacao de logs do guard."""
    stale = [key for key in _LOGGED_REGIME_GUARD_CYCLES if key < int(current_cycle_id) - 100]
    for key in stale:
        _LOGGED_REGIME_GUARD_CYCLES.pop(key, None)


def entry_prob(entry: dict[str, Any]) -> float:
    """Retorna probabilidade calibrada limitada do entry de decisao."""
    metrics = entry.get("metrics") or {}
    raw = metrics.get("calibrated_prob", metrics.get("raw_prob"))
    if raw is None:
        return 0.5
    return max(0.0, min(1.0, float(raw)))


def cross_prob_delta_mean(metrics: dict[str, Any], infra_cfg: dict | None) -> float:
    """Retorna media historica do spread cross-symbol para comparacao de expansao."""
    stored = metrics.get("cross_symbol_prob_delta_mean")
    if stored is not None:
        return float(stored)
    infra = (infra_cfg or {}).get("meta_classifier") or {}
    manifest = infra.get("cross_symbol_prob_delta_mean")
    if manifest is not None:
        return float(manifest)
    return 0.0


def peer_payoff_edge(peer_entry: dict[str, Any] | None, metrics: dict[str, Any]) -> float:
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
    bull_call = entry_prob(bull_entry)
    bear_put = 1.0 - entry_prob(bear_entry)
    metrics["bear_prob_put"] = float(bear_put)
    delta = cross_symbol_conviction_spread(metrics)
    mean = cross_prob_delta_mean(metrics, infra_cfg)
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
    bear_put = 1.0 - entry_prob(bear_entry)
    bull_call = entry_prob(bull_entry)
    metrics["bull_prob_call"] = float(bull_call)
    delta = cross_symbol_conviction_spread(metrics)
    mean = cross_prob_delta_mean(metrics, infra_cfg)
    return bull_call + 1e-12 > bear_put or delta > mean + 1e-12

"""Features cross-symbol de arbitragem para o meta-classificador tabular."""

from __future__ import annotations

from typing import Any

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.meta_classifier_flow_features import FLOW_FEATURE_COUNT
from src.domain.symbols.drift_symbols import DEFAULT_ANCHOR, hedge_peer


CROSS_SYMBOL_FEATURE_COUNT = 3
META_FEATURE_DIM = FEATURE_DIM + CROSS_SYMBOL_FEATURE_COUNT + FLOW_FEATURE_COUNT + 4
ANCHOR_BULL = DEFAULT_ANCHOR
ANCHOR_BEAR = DEFAULT_ANCHOR

CROSS_SYMBOL_KEYS = (
    "cross_symbol_prob_delta",
    "cross_symbol_vol_ratio_diff",
    "cross_symbol_rsi_spread",
)


def _metric_prob(metrics: dict[str, Any]) -> float:
    """Retorna probabilidade calibrada limitada ao intervalo unitario."""
    raw = metrics.get("calibrated_prob", metrics.get("raw_prob"))
    if raw is None:
        return 0.5
    return max(0.0, min(1.0, float(raw)))


def _indicator_value(metrics: dict[str, Any], key: str, *, micro: bool = False) -> float:
    """Le valor indicador do bucket micro ou macro com fallback para indicators."""
    bucket = "micro_indicators" if micro else "indicators"
    chunk = metrics.get(bucket)
    if not isinstance(chunk, dict):
        chunk = metrics["indicators"] if micro and isinstance(metrics.get("indicators"), dict) else {}
    return float(chunk.get(key, 0.0))


def compute_cross_symbol_triplet(
    bull_metrics: dict[str, Any] | None,
    bear_metrics: dict[str, Any] | None,
) -> dict[str, float]:
    """Calcula deltas de arbitragem entre pares; zeros quando nao ha peer."""
    if not isinstance(bull_metrics, dict) or not isinstance(bear_metrics, dict):
        return dict.fromkeys(CROSS_SYMBOL_KEYS, 0.0)
    if bull_metrics is bear_metrics:
        return dict.fromkeys(CROSS_SYMBOL_KEYS, 0.0)
    bull_call = _metric_prob(bull_metrics)
    bear_put = 1.0 - _metric_prob(bear_metrics)
    bull_vol = _indicator_value(bull_metrics, "vol_ratio", micro=True)
    bear_vol = _indicator_value(bear_metrics, "vol_ratio", micro=True)
    bull_rsi = _indicator_value(bull_metrics, "rsi", micro=True)
    bear_rsi = _indicator_value(bear_metrics, "rsi", micro=True)
    return {
        "cross_symbol_prob_delta": abs(bull_call - bear_put),
        "cross_symbol_vol_ratio_diff": bull_vol - bear_vol,
        "cross_symbol_rsi_spread": bull_rsi - bear_rsi,
    }


def attach_cross_symbol_features_to_decisions(decisions: dict[str, dict]) -> None:
    """Propaga triplet cross-symbol para cada decisao antes do prefetch meta."""
    symbols = [str(symbol) for symbol in decisions]
    primary = next((symbol for symbol in symbols if hedge_peer(symbol) is not None), None)
    if primary is None:
        triplet = dict.fromkeys(CROSS_SYMBOL_KEYS, 0.0)
    else:
        peer = hedge_peer(primary)
        primary_entry = decisions.get(primary) if isinstance(decisions.get(primary), dict) else None
        peer_entry = decisions.get(peer) if peer and isinstance(decisions.get(peer), dict) else None
        primary_metrics = primary_entry.get("metrics") if isinstance(primary_entry, dict) else None
        peer_metrics = peer_entry.get("metrics") if isinstance(peer_entry, dict) else None
        triplet = compute_cross_symbol_triplet(
            primary_metrics if isinstance(primary_metrics, dict) else None,
            peer_metrics if isinstance(peer_metrics, dict) else None,
        )
    for entry in decisions.values():
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            continue
        metrics["cross_symbol_features"] = dict(triplet)


def cross_symbol_triplet_from_metrics(metrics: dict[str, Any]) -> list[float]:
    """Extrai triplet cross-symbol previamente anexado em metrics."""
    chunk = metrics.get("cross_symbol_features")
    if isinstance(chunk, dict):
        return [float(chunk.get(key, 0.0)) for key in CROSS_SYMBOL_KEYS]
    return [0.0] * CROSS_SYMBOL_FEATURE_COUNT

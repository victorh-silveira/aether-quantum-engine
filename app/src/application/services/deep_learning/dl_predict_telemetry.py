"""Telemetria micro e bundle cross-symbol para meta-classificador."""

from __future__ import annotations

from typing import Any

from src.application.services.deep_learning.dl_feature_build import precompute_price_series
from src.application.services.meta_classifier_cross_symbol import attach_cross_symbol_features_to_decisions
from src.application.services.meta_classifier_flow_features import flow_features_from_micro_series


def _series_last(series: dict, key: str, default: float = 0.0) -> float:
    """Retorna o ultimo valor finito de uma serie ou default."""
    chunk = series.get(key)
    if chunk is None or len(chunk) == 0:
        return float(default)
    return float(chunk[-1])


def stamp_macro_frame_telemetry(orch: Any, symbol: str, metrics: dict[str, Any], params: dict[str, Any]) -> None:
    """Anexa indicadores da serie MACRO real (300s), distinto do TCN micro."""
    stream = getattr(orch, "stream", None)
    if stream is None or not hasattr(stream, "get_numpy_series"):
        return
    closes = stream.get_numpy_series(str(symbol), "close")
    if closes is None or len(closes) < 8:
        return
    macro_gran = int(params.get("granularity", getattr(stream, "macro_granularity", 300)) or 300)
    series = precompute_price_series(closes, granularity=macro_gran, symbol=str(symbol))
    metrics["macro_indicators"] = {
        "rsi": _series_last(series, "rsi"),
        "vol_ratio": _series_last(series, "vol_ratio_short_long"),
        "adx": _series_last(series, "adx"),
        "hurst": _series_last(series, "hurst"),
    }


def stamp_micro_frame_telemetry(orch: Any, symbol: str, metrics: dict[str, Any], params: dict[str, Any]) -> None:
    """Anexa telemetria micro, fluxo de ticks e desvio Keltner para meta-classificador."""
    stream = getattr(orch, "stream", None)
    if stream is None or not hasattr(stream, "get_micro_numpy_series"):
        return
    closes = stream.get_micro_numpy_series(str(symbol), "close")
    if closes is None or len(closes) < 8:
        return
    micro_gran = int(params.get("micro_granularity", 60))
    high = stream.get_micro_numpy_series(str(symbol), "high")
    low = stream.get_micro_numpy_series(str(symbol), "low")
    open_ = stream.get_micro_numpy_series(str(symbol), "open")
    series = precompute_price_series(closes, granularity=micro_gran, symbol=str(symbol))
    metrics["micro_indicators"] = {
        "rsi": _series_last(series, "rsi"),
        "vol_ratio": _series_last(series, "vol_ratio_short_long"),
    }
    flow = flow_features_from_micro_series(
        closes,
        granularity=micro_gran,
        symbol=str(symbol),
        open_=open_,
        high=high,
        low=low,
    )
    flow["micro_bid_ask_spread_momentum"] = _series_last(series, "micro_bid_ask_spread_momentum")
    flow["micro_bid_ask_spread_momentum_zscore"] = _series_last(series, "micro_bid_ask_spread_momentum_zscore")
    flow["volatility_shadow_ratio"] = _series_last(series, "volatility_shadow_ratio")
    flow["volatility_shadow_ratio_zscore"] = _series_last(series, "volatility_shadow_ratio_zscore")
    tick_buffer = getattr(stream, "tick_buffer", None)
    if tick_buffer is not None and hasattr(tick_buffer, "live_tick_acceleration"):
        flow["micro_tick_acceleration"] = float(tick_buffer.live_tick_acceleration(str(symbol)))
    metrics["flow_features"] = flow
    stamp_macro_frame_telemetry(orch, symbol, metrics, params)


def prepare_meta_classifier_cross_symbol_bundle(
    orch: Any,
    decisions: dict[str, dict],
    params: dict[str, Any],
) -> None:
    """Centraliza telemetria micro paralela e spreads cross-symbol antes do prefetch meta."""
    for symbol, entry in decisions.items():
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            continue
        stamp_micro_frame_telemetry(orch, str(symbol), metrics, params)
    attach_cross_symbol_features_to_decisions(decisions)

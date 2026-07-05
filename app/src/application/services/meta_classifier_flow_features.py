"""Features de velocidade de fluxo micro para o meta-classificador tabular."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.application.services.deep_learning.dl_feature_build import precompute_price_series


FLOW_FEATURE_COUNT = 2
FLOW_FEATURE_KEYS = (
    "micro_tick_acceleration",
    "keltner_deviation_ratio",
)


def compute_keltner_deviation_ratio(
    close: float,
    *,
    ema: float,
    upper: float,
    lower: float,
) -> float:
    """Distancia fracionaria do ultimo preco a linha central do canal Keltner."""
    mid = float(ema)
    if abs(mid) <= 1e-12:
        band = max(float(upper) - float(lower), 1e-12)
        return float((float(close) - mid) / band)
    return float((float(close) - mid) / abs(mid))


def flow_features_from_micro_series(
    closes: np.ndarray,
    *,
    granularity: int,
    symbol: str,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
) -> dict[str, float]:
    """Deriva aceleracao de ticks proxy e desvio Keltner a partir de OHLC micro."""
    if closes is None or len(closes) < 8:
        return dict.fromkeys(FLOW_FEATURE_KEYS, 0.0)
    series = precompute_price_series(
        closes,
        granularity=granularity,
        symbol=symbol,
        open_=open_,
        high=high,
        low=low,
    )
    keltner_pct = float(series["keltner_pct_b"][-1]) if len(series.get("keltner_pct_b", [])) > 0 else 0.5
    keltner_deviation_ratio = keltner_pct - 0.5
    tail = closes[-6:] if len(closes) >= 6 else closes
    deltas = np.diff(tail.astype(np.float64))
    if len(deltas) >= 2:
        v1 = deltas[:-1]
        v2 = deltas[1:]
        micro_tick_acceleration = float(np.mean(v2 - v1))
    else:
        micro_tick_acceleration = 0.0
    return {
        "micro_tick_acceleration": micro_tick_acceleration,
        "keltner_deviation_ratio": keltner_deviation_ratio,
    }


def flow_feature_pair_from_metrics(metrics: dict[str, Any]) -> list[float]:
    """Extrai par de features de fluxo previamente anexado em metrics."""
    chunk = metrics.get("flow_features")
    if isinstance(chunk, dict):
        return [float(chunk.get(key, 0.0)) for key in FLOW_FEATURE_KEYS]
    return [0.0] * FLOW_FEATURE_COUNT

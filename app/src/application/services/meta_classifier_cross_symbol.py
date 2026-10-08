"""Features de microestrutura live para o meta-classificador tabular (1HZ75V)."""

from __future__ import annotations

from typing import Any

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.meta_classifier_flow_features import FLOW_FEATURE_COUNT


CROSS_SYMBOL_FEATURE_COUNT = 3
META_FEATURE_DIM = FEATURE_DIM + CROSS_SYMBOL_FEATURE_COUNT + FLOW_FEATURE_COUNT + 4
ANCHOR_BULL = "1HZ75V"
ANCHOR_BEAR = "1HZ75V"

CROSS_SYMBOL_KEYS = (
    "micro_price_velocity",
    "micro_tick_count_norm",
    "implied_vol_centered",
)


def _clip3(value: float) -> float:
    """Projeta valor no intervalo [-3, 3]."""
    return max(-3.0, min(3.0, float(value)))


def _clip01(value: float) -> float:
    """Projeta valor no intervalo [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def _flow_float(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Le float do bloco flow_features."""
    chunk = metrics.get("flow_features")
    if isinstance(chunk, dict) and chunk.get(key) is not None:
        try:
            return float(chunk[key])
        except (TypeError, ValueError):
            return float(default)
    return float(default)


def _indicator_value(metrics: dict[str, Any], key: str, *, micro: bool = False) -> float:
    """Le valor indicador do bucket micro ou macro com fallback para indicators."""
    bucket = "micro_indicators" if micro else "indicators"
    chunk = metrics.get(bucket)
    if not isinstance(chunk, dict):
        chunk = metrics["indicators"] if micro and isinstance(metrics.get("indicators"), dict) else {}
    try:
        return float(chunk.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def compute_cross_symbol_triplet(
    bull_metrics: dict[str, Any] | None,
    bear_metrics: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Monta triplet de microestrutura do simbolo (dims ex-cross zeradas no 1HZ75V)."""
    metrics = bull_metrics if isinstance(bull_metrics, dict) else bear_metrics
    if not isinstance(metrics, dict):
        return dict.fromkeys(CROSS_SYMBOL_KEYS, 0.0)
    vel = _flow_float(metrics, "price_velocity", 0.0)
    if abs(vel) <= 1e-12:
        vel = _indicator_value(metrics, "price_velocity", micro=True)
    ticks = _flow_float(metrics, "tick_count", 0.0)
    if ticks <= 1e-12:
        ticks = _indicator_value(metrics, "tick_count", micro=True)
    implied = _indicator_value(metrics, "implied_vol_ratio")
    if abs(implied) <= 1e-12:
        implied = _indicator_value(metrics, "implied_vol_ratio", micro=True)
    if abs(implied) <= 1e-12:
        implied = 1.0
    return {
        "micro_price_velocity": _clip3(vel),
        "micro_tick_count_norm": _clip01(ticks / 300.0),
        "implied_vol_centered": _clip3(implied - 1.0),
    }


def attach_cross_symbol_features_to_decisions(decisions: dict[str, dict]) -> None:
    """Anexa triplet de microestrutura live em cada decisao antes do prefetch meta."""
    for entry in decisions.values():
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            continue
        metrics["cross_symbol_features"] = compute_cross_symbol_triplet(metrics)


def cross_symbol_triplet_from_metrics(metrics: dict[str, Any]) -> list[float]:
    """Extrai triplet de microestrutura previamente anexado em metrics."""
    chunk = metrics.get("cross_symbol_features")
    if isinstance(chunk, dict):
        return [float(chunk.get(key, 0.0)) for key in CROSS_SYMBOL_KEYS]
    return [0.0] * CROSS_SYMBOL_FEATURE_COUNT

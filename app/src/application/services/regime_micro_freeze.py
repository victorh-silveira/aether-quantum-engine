"""Freeze/squeeze de regime micro; sem inversao CALL/PUT."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.bb_width_adaptive_squeeze import (
    BB_WIDTH_ANOMALY_RATIO,
    evaluate_bb_width_squeeze,
    harmonic_mean_bb_width,
)
from src.domain.risk.stake_sizing import metric_float


CHOP_CONGESTION_Z_EDGE = 0.20
TICK_ACCEL_NEUTRAL_EPS = 0.01
REGIME_CHOP_CONGESTION = "CHOP_CONGESTION"
SIGNAL_SUSPENDED = "SIGNAL_SUSPENDED"

_SQUEEZE_LOGGER = logging.getLogger("AETH")


def _read_micro_bb_width(metrics: dict[str, Any]) -> float | None:
    """Le bb_width priorizando micro, indicators e macro."""
    micro = metrics.get("micro_indicators")
    if isinstance(micro, dict) and micro.get("bb_width") is not None:
        return float(micro["bb_width"])
    indicators = metrics.get("indicators") or {}
    if indicators.get("bb_width") is not None:
        return float(indicators["bb_width"])
    macro = metrics.get("macro_indicators") or {}
    if macro.get("bb_width") is not None:
        return float(macro["bb_width"])
    return None


def _read_micro_tick_acceleration(metrics: dict[str, Any]) -> float:
    """Le aceleracao de ticks do bloco flow_features."""
    flow = metrics.get("flow_features")
    if isinstance(flow, dict) and flow.get("micro_tick_acceleration") is not None:
        return float(flow["micro_tick_acceleration"])
    return 0.0


def severe_bb_compression(metrics: dict[str, Any]) -> bool:
    """True quando a largura das bandas indica compressao anomala."""
    bb_width = _read_micro_bb_width(metrics)
    if bb_width is None:
        return False
    configured_ratio = metrics.get("bb_width_anomaly_ratio")
    ratio = float(configured_ratio) if configured_ratio is not None else BB_WIDTH_ANOMALY_RATIO
    compressed, mean, width = evaluate_bb_width_squeeze(bb_width, anomaly_ratio=ratio)
    metrics["bb_width_harmonic_mean"] = float(mean)
    metrics["bb_width_anomaly_ratio"] = float(ratio)
    metrics["bb_width_current"] = float(width)
    metrics["bb_width_anomalous_compression"] = bool(compressed)
    return compressed


def micro_volatility_squeeze_active(metrics: dict[str, Any]) -> bool:
    """True sob compressao BB ou desaceleracao de ticks."""
    tick_accel = _read_micro_tick_acceleration(metrics)
    return severe_bb_compression(metrics) or tick_accel < 0.0


def chop_congestion_regime_active(metrics: dict[str, Any], *, persistence_filter_active: bool) -> bool:
    """True quando o filtro detecta congestao CHOP."""
    if not persistence_filter_active:
        return False
    z_edge = abs(float(metrics.get("edge_zscore", 0.0)))
    tick_accel = abs(_read_micro_tick_acceleration(metrics))
    return z_edge + 1e-12 < CHOP_CONGESTION_Z_EDGE and tick_accel <= TICK_ACCEL_NEUTRAL_EPS


def apply_regime_freeze_if_congested(metrics: dict[str, Any], *, persistence_filter_active: bool) -> bool:
    """Congela o ciclo sob congestao micro sem inverter lado."""
    if not chop_congestion_regime_active(metrics, persistence_filter_active=persistence_filter_active):
        return False
    metrics["regime_classification"] = REGIME_CHOP_CONGESTION
    metrics["regime_guard_action"] = "FREEZE: SKIP CYCLE"
    metrics["signal_status"] = SIGNAL_SUSPENDED
    return True


def log_d_squeeze_audit(symbol: str | None, metrics: dict[str, Any]) -> None:
    """Emite auditoria D-SQUEEZE para diagnostico de squeeze."""
    bb_width = _read_micro_bb_width(metrics)
    harmonic = float(metrics.get("bb_width_harmonic_mean", harmonic_mean_bb_width()))
    _SQUEEZE_LOGGER.info(
        "[D-SQUEEZE] %s bb_width=%.4f harm_mean=%.4f ratio=%.2f tick_accel=%.4f payoff=%.4f score=%.2f",
        symbol or "?",
        float(bb_width) if bb_width is not None else 0.0,
        harmonic,
        float(BB_WIDTH_ANOMALY_RATIO),
        _read_micro_tick_acceleration(metrics),
        float(metrics.get("predicted_payoff_edge", metrics.get("meta_calibrated_payoff_score", 0.0))),
        metric_float(metrics, "trade_score", default=0.0),
    )

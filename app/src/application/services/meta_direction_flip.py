"""Inversao direcional orientada pelo meta-classificador em exaustao micro."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.bb_width_adaptive_squeeze import (
    BB_WIDTH_ANOMALY_RATIO,
    evaluate_bb_width_squeeze,
    harmonic_mean_bb_width,
)
from src.application.services.execution_quality_gate import sync_direction_margin
from src.domain.models.trade import TradeDirection
from src.domain.risk.stake_sizing import metric_float


META_FLIP_PAYOFF_THRESHOLD_BASE = 0.42
META_FLIP_PAYOFF_THRESHOLD_SQUEEZE = 0.49
META_FLIP_TRADE_SCORE = 0.75
META_FLIP_SQUEEZE_TRADE_SCORE = 0.52
CHOP_CONGESTION_Z_EDGE = 0.20
TICK_ACCEL_NEUTRAL_EPS = 0.01
REGIME_CHOP_CONGESTION = "CHOP_CONGESTION"
SIGNAL_SUSPENDED = "SIGNAL_SUSPENDED"

_SQUEEZE_LOGGER = logging.getLogger("AETH")


def _read_micro_bb_width(metrics: dict[str, Any]) -> float | None:
    """Le bb_width micro M5 com fallback para indicadores macro anexados."""
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
    """Le aceleracao de ticks micro do par flow_features."""
    flow = metrics.get("flow_features")
    if isinstance(flow, dict) and flow.get("micro_tick_acceleration") is not None:
        return float(flow["micro_tick_acceleration"])
    return 0.0


def severe_bb_compression(metrics: dict[str, Any]) -> bool:
    """Indica compressao anomala de bb_width abaixo do ratio elastico da media harmonica movel."""
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
    """Indica squeeze M5 por bb_width comprimido ou desaceleracao institucional de ticks."""
    tick_accel = _read_micro_tick_acceleration(metrics)
    return severe_bb_compression(metrics) or tick_accel < 0.0


def chop_congestion_regime_active(metrics: dict[str, Any], *, persistence_filter_active: bool) -> bool:
    """Indica congestao micro com edge colapsado e aceleracao de ticks neutra."""
    if not persistence_filter_active:
        return False
    z_edge = abs(float(metrics.get("edge_zscore", 0.0)))
    tick_accel = abs(_read_micro_tick_acceleration(metrics))
    return z_edge + 1e-12 < CHOP_CONGESTION_Z_EDGE and tick_accel <= TICK_ACCEL_NEUTRAL_EPS


def apply_regime_freeze_if_congested(metrics: dict[str, Any], *, persistence_filter_active: bool) -> bool:
    """Classifica CHOP_CONGESTION e sinaliza suspensao quando micro sinais conflitam."""
    if not chop_congestion_regime_active(metrics, persistence_filter_active=persistence_filter_active):
        return False
    metrics["regime_classification"] = REGIME_CHOP_CONGESTION
    metrics["regime_guard_action"] = "FREEZE: SKIP CYCLE"
    metrics["signal_status"] = SIGNAL_SUSPENDED
    return True


def resolve_dynamic_flip_threshold(metrics: dict[str, Any]) -> tuple[float, bool]:
    """Retorna limiar elastico de payoff e flag de squeeze micro ativo."""
    squeeze_active = micro_volatility_squeeze_active(metrics)
    threshold = META_FLIP_PAYOFF_THRESHOLD_SQUEEZE if squeeze_active else META_FLIP_PAYOFF_THRESHOLD_BASE
    return threshold, squeeze_active


def should_flip_direction(
    _dl_dir: TradeDirection,
    payoff_score: float,
    *,
    meta_applied: bool,
    flip_threshold: float = META_FLIP_PAYOFF_THRESHOLD_BASE,
    polarity_inverted: bool = False,
) -> bool:
    """Indica inversao quando payoff micro sinaliza saturacao severa de topo/fundo."""
    if polarity_inverted or not meta_applied:
        return False
    return float(payoff_score) < float(flip_threshold)


def flipped_direction(dl_dir: TradeDirection) -> TradeDirection:
    """Retorna direcao oposta ao sinal TCN."""
    return TradeDirection.PUT if dl_dir == TradeDirection.CALL else TradeDirection.CALL


def invert_execution_direction_enabled(exec_cfg: dict[str, Any] | None) -> bool:
    """Indica se a polaridade de execucao deve inverter CALL/PUT."""
    if not isinstance(exec_cfg, dict):
        return False
    return bool(exec_cfg.get("invert_execution_direction", False))


def apply_configured_direction_invert(
    calibrated_prob: float,
    raw_prob: float,
    direction: TradeDirection | None,
    *,
    exec_cfg: dict[str, Any] | None,
) -> tuple[float, float, TradeDirection | None, bool]:
    """Inverte probabilidade e direcao quando invert_execution_direction esta ativo."""
    if not invert_execution_direction_enabled(exec_cfg):
        return float(calibrated_prob), float(raw_prob), direction, False
    cal = 1.0 - float(calibrated_prob)
    raw = 1.0 - float(raw_prob)
    flipped = flipped_direction(direction) if direction is not None else None
    if flipped is None:
        flipped = TradeDirection.CALL if cal + 1e-12 >= 0.5 else TradeDirection.PUT
    return cal, raw, flipped, True


def log_d_squeeze_audit(symbol: str | None, metrics: dict[str, Any]) -> None:
    """Emite log [D-SQUEEZE] com metricas de compressao micro para auditoria."""
    bb_width = _read_micro_bb_width(metrics)
    harmonic = float(metrics.get("bb_width_harmonic_mean", harmonic_mean_bb_width()))
    _SQUEEZE_LOGGER.info(
        "[D-SQUEEZE] %s bb_width=%.4f harm_mean=%.4f ratio=%.2f tick_accel=%.4f payoff=%.4f threshold=%.2f flip=%s score=%.2f",
        symbol or "?",
        float(bb_width) if bb_width is not None else 0.0,
        harmonic,
        float(BB_WIDTH_ANOMALY_RATIO),
        _read_micro_tick_acceleration(metrics),
        float(metrics.get("predicted_payoff_edge", metrics.get("meta_calibrated_payoff_score", 0.0))),
        float(metrics.get("dynamic_flip_threshold", META_FLIP_PAYOFF_THRESHOLD_BASE)),
        bool(metrics.get("meta_direction_flip", False)),
        metric_float(metrics, "trade_score", default=0.0),
    )


def apply_meta_direction_flip(
    dl_dir: TradeDirection,
    metrics: dict[str, Any],
    payoff_score: float,
    *,
    meta_applied: bool,
    tcn_probability: float,
) -> tuple[TradeDirection, float]:
    """Aplica matriz de inversao por payoff elastico e atualiza metricas de auditoria."""
    flip_threshold, squeeze_active = resolve_dynamic_flip_threshold(metrics)
    metrics["dynamic_flip_threshold"] = float(flip_threshold)
    metrics["meta_squeeze_active"] = bool(squeeze_active)
    if not should_flip_direction(
        dl_dir,
        payoff_score,
        meta_applied=meta_applied,
        flip_threshold=flip_threshold,
        polarity_inverted=bool(metrics.get("execution_direction_invert")),
    ):
        return dl_dir, float(payoff_score)
    exec_dir = flipped_direction(dl_dir)
    score = META_FLIP_SQUEEZE_TRADE_SCORE if squeeze_active else META_FLIP_TRADE_SCORE
    metrics["meta_calibrated_payoff_score"] = float(payoff_score)
    metrics["meta_classifier_applied"] = bool(meta_applied)
    metrics["meta_direction_flip"] = True
    metrics["meta_squeeze_flip"] = bool(squeeze_active)
    metrics["direction_inverted"] = True
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
    _ = tcn_probability
    return exec_dir, score

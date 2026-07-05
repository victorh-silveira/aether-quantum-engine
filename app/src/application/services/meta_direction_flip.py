"""Inversao direcional orientada pelo meta-classificador em exaustao micro."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection


META_FLIP_PAYOFF_THRESHOLD = 0.42
META_FLIP_TRADE_SCORE = 0.75


def should_flip_direction(
    _dl_dir: TradeDirection,
    payoff_score: float,
    *,
    meta_applied: bool,
) -> bool:
    """Indica inversao quando payoff micro sinaliza saturacao severa de topo/fundo."""
    if not meta_applied:
        return False
    return float(payoff_score) < META_FLIP_PAYOFF_THRESHOLD


def flipped_direction(dl_dir: TradeDirection) -> TradeDirection:
    """Retorna direcao oposta ao sinal TCN."""
    return TradeDirection.PUT if dl_dir == TradeDirection.CALL else TradeDirection.CALL


def apply_meta_direction_flip(
    dl_dir: TradeDirection,
    metrics: dict[str, Any],
    payoff_score: float,
    *,
    meta_applied: bool,
    tcn_probability: float,
) -> tuple[TradeDirection, float]:
    """Aplica matriz de inversao por probabilidade e atualiza metricas de auditoria."""
    if not should_flip_direction(dl_dir, payoff_score, meta_applied=meta_applied):
        return dl_dir, float(payoff_score)
    exec_dir = flipped_direction(dl_dir)
    score = META_FLIP_TRADE_SCORE
    metrics["meta_calibrated_payoff_score"] = float(payoff_score)
    metrics["meta_classifier_applied"] = bool(meta_applied)
    metrics["meta_direction_flip"] = True
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
    metrics["direction_margin"] = abs(metrics["direction_call_score"] - metrics["direction_put_score"])
    _ = tcn_probability
    return exec_dir, score

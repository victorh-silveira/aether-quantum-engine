"""Facade do barramento universal de regimes CALL/PUT."""

from __future__ import annotations

import math
from typing import Any

from src.application.services.execution_universal_regime_evaluator import (
    UniversalRegimeEvaluator,
    apply_regime_direction_boost,
    invert_trade_direction,
)
from src.application.services.execution_universal_regime_types import (
    COMPRESSION_TRAP_SCORE,
    RegimeEvaluation,
    RegimeState,
    UniversalLossEvaluation,
    regime_evaluation_to_legacy,
)
from src.domain.models.trade import TradeDirection


__all__ = [
    "COMPRESSION_TRAP_SCORE",
    "RegimeEvaluation",
    "UniversalLossEvaluation",
    "UniversalRegimeEvaluator",
    "apply_compression_trap_boost",
    "apply_regime_direction_boost",
    "apply_universal_regime_penalty_to_metrics",
    "apply_universal_regime_resolution",
    "evaluate_universal_loss_scenarios",
    "invert_trade_direction",
    "log_regime_audit",
    "map_volatility_regime",
    "map_volatility_regime_to_metrics",
    "regime_skip_blocks_trade",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Converte indicador para float finito com fallback defensivo."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _indicators(metrics: dict) -> dict:
    """Extrai indicadores macro M15 das metricas (fallback para bloco legado)."""
    macro = metrics.get("macro_indicators")
    if isinstance(macro, dict) and macro:
        return macro
    raw = metrics.get("indicators")
    return raw if isinstance(raw, dict) else {}


def regime_skip_blocks_trade(metrics: dict) -> bool:
    """Indica se o barramento de regime bloqueou execucao no ciclo."""
    return bool(metrics.get("regime_skip_cycle"))


def map_volatility_regime(metrics: dict) -> str:
    """Classifica compressao, neutro ou expansao via vol_ratio."""
    vol_ratio = _safe_float(_indicators(metrics).get("vol_ratio"), 1.0)
    if vol_ratio < 0.85:
        return "compression"
    if vol_ratio > 1.15:
        return "expansion"
    return "neutral"


def map_volatility_regime_to_metrics(metrics: dict) -> str:
    """Persiste regime de volatilidade nas metricas do candidato."""
    regime = map_volatility_regime(metrics)
    metrics["vol_regime"] = regime
    return regime


def apply_compression_trap_boost(metrics: dict, exec_dir: TradeDirection, score: float) -> None:
    """Eleva score e marca inversao tatica em regime COMPRESSION_TRAP."""
    apply_regime_direction_boost(metrics, exec_dir, score, "COMPRESSION_TRAP")


def apply_universal_regime_penalty_to_metrics(metrics: dict, evaluation: UniversalLossEvaluation) -> float:
    """Aplica atenuacao universal ao trade_score e scores direcionais."""
    if evaluation.scenario is None or evaluation.score_factor >= 1.0:
        return 1.0
    factor = float(evaluation.score_factor)
    metrics["universal_regime_scenario"] = evaluation.scenario
    metrics["universal_regime"] = evaluation.scenario
    metrics["universal_regime_score_factor"] = factor
    if evaluation.gate_penalty:
        metrics["gate_penalty"] = evaluation.gate_penalty
    keys = ("trade_score", "conviction", "resolved_conviction", "direction_call_score", "direction_put_score")
    for key in keys:
        if key in metrics and metrics[key] is not None:
            metrics[key] = max(0.0, _safe_float(metrics[key]) * factor)
    if evaluation.mandatory_conviction_floor is not None:
        floor = float(evaluation.mandatory_conviction_floor)
        for key in ("trade_score", "conviction", "resolved_conviction"):
            if key in metrics:
                metrics[key] = floor
    return factor


def apply_universal_regime_resolution(
    metrics: dict,
    evaluation: UniversalLossEvaluation,
    exec_dir: TradeDirection,
) -> TradeDirection:
    """Aplica penalidade ou inversao tatica conforme cenario universal legado."""
    if evaluation.scenario is None:
        return exec_dir
    try:
        regime_state = RegimeState(evaluation.scenario)
    except ValueError:
        return exec_dir
    regime_eval = RegimeEvaluation(
        regime=regime_state,
        direction_inverted=evaluation.invert_direction,
        gate_penalty=evaluation.gate_penalty,
        trap_boost_score=evaluation.trap_boost_score,
        score_factor=evaluation.score_factor,
        mandatory_conviction_floor=evaluation.mandatory_conviction_floor,
    )
    dl_raw = metrics.get("dl_direction")
    try:
        dl_dir = TradeDirection[str(dl_raw).upper()] if dl_raw else exec_dir
    except (KeyError, ValueError):
        dl_dir = exec_dir
    evaluator = UniversalRegimeEvaluator({})
    return evaluator.apply(metrics, regime_eval, exec_dir, dl_dir=dl_dir)


def _dl_direction_from_metrics(metrics: dict) -> TradeDirection | None:
    """Resolve direcao DL persistida nas metricas."""
    raw = metrics.get("dl_direction") or metrics.get("resolved_direction")
    if raw is None:
        return None
    try:
        return TradeDirection[str(raw).upper()]
    except (KeyError, ValueError):
        return None


def evaluate_universal_loss_scenarios(
    metrics: dict,
    *,
    dl_dir: TradeDirection | None = None,
    exec_dir: TradeDirection | None = None,
    mandatory_min_signal: float = 0.56,
    regime_cfg: dict | None = None,
    recovery_active: bool = False,
    continuous_mode: bool = False,
) -> UniversalLossEvaluation:
    """Classifica cenario de loss churn via barramento universal legado."""
    resolved_dl = dl_dir or _dl_direction_from_metrics(metrics)
    resolved_exec = exec_dir if exec_dir is not None else resolved_dl
    if resolved_dl is None or resolved_exec is None:
        return UniversalLossEvaluation(scenario=None, score_factor=1.0)
    evaluator = UniversalRegimeEvaluator(
        regime_cfg,
        recovery_active=recovery_active,
        continuous_mode=continuous_mode,
        mandatory_min_signal=mandatory_min_signal,
    )
    evaluation = evaluator.evaluate(metrics, dl_dir=resolved_dl, exec_dir=resolved_exec)
    return regime_evaluation_to_legacy(evaluation)


def log_regime_audit(
    logger,
    cid: str,
    symbol: str,
    dl_dir: TradeDirection,
    exec_dir: TradeDirection,
    metrics: dict,
    *,
    recovery_active: bool = False,
) -> None:
    """Registra log de auditoria do regime macro com execucao micro M1."""
    _ = dl_dir
    regime = metrics.get("universal_regime")
    if not regime:
        return
    stake_mode = "D'ALEMBERT" if recovery_active else "KELLY"
    logger.info(
        "[%s] REGIME MACRO: %s | Execucao Micro: M1 %s | Stake: %s | %s",
        cid,
        regime,
        exec_dir.name,
        stake_mode,
        symbol,
    )

"""Motor de estados macro para chaveamento CALL/PUT por regime de mercado."""

from __future__ import annotations

import math
from typing import Any

from src.application.services.execution_universal_regime_types import (
    RegimeEvaluation,
    RegimeState,
    parse_regime_evaluator_cfg,
)
from src.domain.models.trade import TradeDirection


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


def invert_trade_direction(direction: TradeDirection) -> TradeDirection:
    """Inverte CALL/PUT para reversao tatica."""
    return TradeDirection.PUT if direction == TradeDirection.CALL else TradeDirection.CALL


def apply_regime_direction_boost(
    metrics: dict,
    exec_dir: TradeDirection,
    score: float,
    regime_name: str,
) -> None:
    """Eleva score e marca inversao tatica para regime com boost."""
    boost = float(score)
    metrics["universal_regime"] = regime_name
    metrics["universal_regime_scenario"] = regime_name
    metrics["compression_trap_inverted"] = regime_name == RegimeState.COMPRESSION_TRAP.value
    metrics["direction_inverted"] = True
    metrics["market_decision_score_override"] = boost
    metrics["universal_regime_score_factor"] = 1.0
    for key in ("trade_score", "conviction", "resolved_conviction"):
        metrics[key] = boost
    metrics["exec_direction"] = exec_dir.name
    metrics["resolved_direction"] = exec_dir.name
    if exec_dir == TradeDirection.CALL:
        metrics["direction_call_score"] = boost
        metrics["direction_put_score"] = max(0.0, 1.0 - boost)
    else:
        metrics["direction_put_score"] = boost
        metrics["direction_call_score"] = max(0.0, 1.0 - boost)


class UniversalRegimeEvaluator:
    """Classifica candidatos em quatro regimes e aplica chaveamento CALL/PUT."""

    def __init__(
        self,
        cfg: dict | None,
        *,
        recovery_active: bool = False,
        continuous_mode: bool = False,
        mandatory_min_signal: float = 0.56,
        kelly_cfg: dict | None = None,
    ) -> None:
        """Inicializa avaliador com config, recovery e modo continuo."""
        self._cfg = parse_regime_evaluator_cfg(cfg)
        self._recovery_active = recovery_active
        self._continuous_mode = continuous_mode
        self._mandatory_min_signal = float(mandatory_min_signal)
        self._kelly_cfg = kelly_cfg if isinstance(kelly_cfg, dict) else {}

    @property
    def enabled(self) -> bool:
        """Indica se o barramento de regimes esta ativo."""
        return bool(self._cfg.get("enabled", True))

    def evaluate(
        self,
        metrics: dict,
        *,
        dl_dir: TradeDirection,
        exec_dir: TradeDirection,
    ) -> RegimeEvaluation:
        """Classifica regime com prioridade risco-primeiro."""
        _ = dl_dir
        if not self.enabled:
            return RegimeEvaluation(regime=None)
        indicators = _indicators(metrics)
        adx = _safe_float(indicators.get("adx"), 0.0)
        hurst = _safe_float(indicators.get("hurst"), 0.5)
        vol_ratio = _safe_float(indicators.get("vol_ratio"), 1.0)
        rsi = _safe_float(indicators.get("rsi"), 0.5)
        cmo = _safe_float(indicators.get("cmo"), 0.0)
        cfg = self._cfg
        trend_adx = float(cfg["trend_adx_min"])
        climax_rsi = float(cfg["climax_rsi_max"])
        climax_cmo = float(cfg["climax_cmo_min"])
        if adx >= trend_adx and (rsi > climax_rsi or rsi < 1.0 - climax_rsi) and abs(cmo) >= climax_cmo:
            return RegimeEvaluation(
                regime=RegimeState.CLIMAX_EXHAUSTION,
                direction_inverted=True,
                trap_boost_score=float(cfg["climax_boost_score"]),
            )
        comp_adx = float(cfg["compression_adx_max"])
        comp_hurst = float(cfg["compression_hurst_max"])
        comp_vol = float(cfg["compression_vol_max"])
        if adx < comp_adx and hurst <= comp_hurst and vol_ratio < comp_vol:
            stretched = (exec_dir == TradeDirection.CALL and rsi > float(cfg["compression_rsi_call_min"])) or (
                exec_dir == TradeDirection.PUT and rsi < float(cfg["compression_rsi_put_max"])
            )
            return RegimeEvaluation(
                regime=RegimeState.COMPRESSION_TRAP,
                direction_inverted=stretched,
                trap_boost_score=float(cfg["compression_boost_score"]) if stretched else None,
            )
        if adx >= trend_adx and hurst > float(cfg["trend_hurst_min"]) and vol_ratio >= float(cfg["trend_vol_min"]):
            return RegimeEvaluation(regime=RegimeState.TREND_EXPANSION)
        call_votes = int(_safe_float(metrics.get("call_votes"), 0))
        put_votes = int(_safe_float(metrics.get("put_votes"), 0))
        entropic_hurst = float(cfg["entropic_hurst_max"])
        votes_tied = call_votes == put_votes and call_votes > 0
        if votes_tied or hurst < entropic_hurst:
            return RegimeEvaluation(
                regime=RegimeState.ENTROPIC_NOISE,
                gate_penalty="noise",
                regime_skip_cycle=True,
                score_factor=1.0,
                mandatory_conviction_floor=None,
            )
        return RegimeEvaluation(regime=None)

    def apply(
        self,
        metrics: dict,
        evaluation: RegimeEvaluation,
        exec_dir: TradeDirection,
        *,
        dl_dir: TradeDirection,
    ) -> TradeDirection:
        """Persiste regime nas metricas e retorna direcao final de execucao."""
        if evaluation.regime is None:
            return exec_dir
        regime_name = evaluation.regime.value
        metrics["universal_regime"] = regime_name
        metrics["universal_regime_scenario"] = regime_name
        if evaluation.gate_penalty:
            metrics["gate_penalty"] = evaluation.gate_penalty
        if evaluation.regime_skip_cycle:
            metrics["regime_skip_cycle"] = True
            return exec_dir
        if evaluation.regime == RegimeState.TREND_EXPANSION:
            metrics["direction_inverted"] = False
            metrics["universal_regime_score_factor"] = 1.0
            metrics["exec_direction"] = dl_dir.name
            metrics["resolved_direction"] = dl_dir.name
            return dl_dir
        if evaluation.direction_inverted and evaluation.trap_boost_score is not None:
            if evaluation.regime == RegimeState.CLIMAX_EXHAUSTION:
                inverted = invert_trade_direction(dl_dir)
            else:
                inverted = invert_trade_direction(exec_dir)
            apply_regime_direction_boost(metrics, inverted, evaluation.trap_boost_score, regime_name)
            return inverted
        if evaluation.regime == RegimeState.COMPRESSION_TRAP and not evaluation.direction_inverted:
            metrics["universal_regime_score_factor"] = 1.0
        return exec_dir

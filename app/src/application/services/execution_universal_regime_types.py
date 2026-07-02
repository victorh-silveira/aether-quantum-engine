"""Tipos e constantes do barramento de regimes universais CALL/PUT."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RegimeState(StrEnum):
    """Estados macro de mercado para chaveamento direcional."""

    TREND_EXPANSION = "TREND_EXPANSION"
    COMPRESSION_TRAP = "COMPRESSION_TRAP"
    CLIMAX_EXHAUSTION = "CLIMAX_EXHAUSTION"
    ENTROPIC_NOISE = "ENTROPIC_NOISE"


COMPRESSION_TRAP_SCORE = 0.75
CLIMAX_EXHAUSTION_SCORE = 0.76

_DEFAULT_REGIME_CFG: dict[str, float | bool] = {
    "enabled": True,
    "trend_adx_min": 0.28,
    "trend_hurst_min": 0.56,
    "trend_vol_min": 1.05,
    "compression_adx_max": 0.18,
    "compression_hurst_max": 0.48,
    "compression_vol_max": 0.80,
    "compression_rsi_call_min": 0.62,
    "compression_rsi_put_max": 0.38,
    "climax_rsi_max": 0.70,
    "climax_cmo_min": 0.52,
    "climax_boost_score": CLIMAX_EXHAUSTION_SCORE,
    "compression_boost_score": COMPRESSION_TRAP_SCORE,
    "entropic_hurst_max": 0.42,
}


@dataclass(frozen=True)
class RegimeEvaluation:
    """Resultado da classificacao de regime universal."""

    regime: RegimeState | None
    direction_inverted: bool = False
    gate_penalty: str | None = None
    regime_skip_cycle: bool = False
    trap_boost_score: float | None = None
    score_factor: float = 1.0
    mandatory_conviction_floor: float | None = None


@dataclass(frozen=True)
class UniversalLossEvaluation:
    """Resultado legado da classificacao de cenario de perda."""

    scenario: str | None
    score_factor: float
    gate_penalty: str | None = None
    mandatory_conviction_floor: float | None = None
    invert_direction: bool = False
    trap_boost_score: float | None = None


def parse_regime_evaluator_cfg(raw: dict | None) -> dict[str, float | bool]:
    """Mescla configuracao de regime com defaults do motor."""
    merged: dict[str, float | bool] = dict(_DEFAULT_REGIME_CFG)
    if isinstance(raw, dict):
        for key in _DEFAULT_REGIME_CFG:
            if key not in raw:
                continue
            if key == "enabled":
                merged[key] = bool(raw[key])
            else:
                merged[key] = float(raw[key])
    return merged


def regime_evaluation_to_legacy(evaluation: RegimeEvaluation) -> UniversalLossEvaluation:
    """Converte avaliacao de regime para formato legado de loss churn."""
    if evaluation.regime is None:
        return UniversalLossEvaluation(scenario=None, score_factor=1.0)
    return UniversalLossEvaluation(
        scenario=evaluation.regime.value,
        score_factor=evaluation.score_factor,
        gate_penalty=evaluation.gate_penalty,
        mandatory_conviction_floor=evaluation.mandatory_conviction_floor,
        invert_direction=evaluation.direction_inverted,
        trap_boost_score=evaluation.trap_boost_score,
    )

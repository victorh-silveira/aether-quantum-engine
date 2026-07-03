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


class MicroPositionZone(StrEnum):
    """Zonas de micro-posicionamento M1 no canal de preco de 60s."""

    UPPER_BOUNDARY = "UPPER_BOUNDARY"
    LOWER_BOUNDARY = "LOWER_BOUNDARY"
    FAIR_VALUE_MIDDLE = "FAIR_VALUE_MIDDLE"


COMPRESSION_TRAP_SCORE = 0.75
CLIMAX_EXHAUSTION_SCORE = 0.76

MICRO_KELTNER_TOPO_TRIGGER = 1.05
MICRO_KELTNER_FUNDO_TRIGGER = -0.05
MICRO_BB_UPPER_TRIGGER = 0.95
MICRO_BB_LOWER_TRIGGER = 0.05
MICRO_BREAKOUT_MOMENTUM_BOOST = 0.05
MICRO_BREAKOUT_SCORE_CAP = 0.95
MICRO_EXHAUSTION_OVERRIDE_SCORE = 0.75
MICRO_MIDDLE_UNCERTAINTY_SCORE = 0.50
MICRO_TREND_ADX_MIN = 0.23
MICRO_TREND_VOL_MIN = 1.0
MICRO_LOW_CONSENSUS_MARGIN = 2
MICRO_LOW_CONSENSUS_MIN_MINORITY = 2
MICRO_MIDDLE_UNCERTAINTY_REASON = "micro_middle_uncertainty_skip"
MICRO_EXHAUSTION_REGIME_TOKEN = "MICRO_EXHAUSTION_TRAP"

_DEFAULT_MICRO_MATRIX_CFG: dict[str, float] = {
    "keltner_topo_trigger": MICRO_KELTNER_TOPO_TRIGGER,
    "keltner_fundo_trigger": MICRO_KELTNER_FUNDO_TRIGGER,
    "breakout_momentum_boost": MICRO_BREAKOUT_MOMENTUM_BOOST,
}

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
class MicroMatrixDecision:
    """Decisao tatica da matriz de micro-posicionamento M1."""

    zone: MicroPositionZone
    breakout_boost: float = 0.0
    exhaustion_invert: bool = False
    score_override: float | None = None
    middle_uncertainty: bool = False


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
    micro: MicroMatrixDecision | None = None


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


def parse_micro_matrix_cfg(raw: dict | None) -> dict[str, float]:
    """Mescla knobs de micro-posicionamento M1 com defaults institucionais."""
    merged: dict[str, float] = dict(_DEFAULT_MICRO_MATRIX_CFG)
    if isinstance(raw, dict):
        for key in _DEFAULT_MICRO_MATRIX_CFG:
            if key in raw:
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

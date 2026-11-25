"""Grade imutavel de pesquisa de frequencia, sem promocao operacional automatica."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from src.indicator.domain import Candle
from src.indicator.strategy_persistent_trend import (
    TOXIC_HOURS_UTC,
    _branch_di_adx,
    _branch_di_continuation,
    _branch_di_pullback,
    _branch_h1_di,
    _candle_dir,
    _resolve_h1_trend,
    calc_wilder_di_adx,
)


DI_SCALES = (0.5, 0.625, 0.75, 0.875, 1.0)
ADX_MINIMA = (20.0, 25.0, 30.0, 35.0)
ALL_BRANCHES = (1, 2, 3, 4, 5)
COMPOSITIONS = (
    *((branch,) for branch in ALL_BRANCHES),
    ALL_BRANCHES,
    *(tuple(branch for branch in ALL_BRANCHES if branch != removed) for removed in ALL_BRANCHES),
)


@dataclass(frozen=True)
class Candidate:
    """Parametros fixos de uma hipotese previamente declarada de pesquisa."""

    branches: tuple[int, ...] = ALL_BRANCHES
    di_scale: float = 1.0
    adx_min: float = 35.0

    def __post_init__(self) -> None:
        """Rejeita composicoes ambiguas e limites nao finitos antes da pesquisa."""
        if (
            not isinstance(self.branches, tuple)
            or not self.branches
            or tuple(sorted(set(self.branches))) != self.branches
            or any(branch not in ALL_BRANCHES for branch in self.branches)
        ):
            raise ValueError("composicao de ramos invalida")
        if not isfinite(self.di_scale) or self.di_scale <= 0 or not isfinite(self.adx_min) or self.adx_min < 0:
            raise ValueError("limiar de pesquisa invalido")

    @property
    def version(self) -> str:
        """Identifica os parametros e o contrato causal da hipotese por hash."""
        spec = {
            "branches": self.branches,
            "di_scale": float(self.di_scale),
            "adx_min": float(self.adx_min),
            "contract": "persistent_v3:di14_20_24_28:h1ema20_50:toxic3_6_19:strict:v1",
        }
        return "frequency-1-" + sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]

    @property
    def label(self) -> str:
        """Descreve a hipotese sem depender do resultado observado."""
        branches = "".join(str(branch) for branch in self.branches)
        return f"b{branches}_di{self.di_scale:g}_adx{self.adx_min:g}"


@dataclass(frozen=True)
class Features:
    """Estatisticas causais compartilhadas entre candidatos no mesmo instante."""

    di14: float = 0.0
    di20: float = 0.0
    di24: float = 0.0
    di28: float = 0.0
    adx20: float = 0.0
    h1_trend: int = 0
    direction: int = 0
    reason: str = ""


def generate_candidates() -> tuple[Candidate, ...]:
    """Expande a grade nominal e deduplica ADX sem efeito nos ramos sem B5."""
    return tuple(
        Candidate(branches, scale, adx)
        for branches in COMPOSITIONS
        for scale in DI_SCALES
        for adx in (ADX_MINIMA if 5 in branches else (35.0,))
    )


def extract_features(bars: list[Candle], h1_bars: list[Candle] | None = None) -> Features:
    """Reutiliza Wilder e H1 da v3 em janela fechada validada pelo chamador."""
    if len(bars) < 51 or any(bar.high == bar.low for bar in bars[-2:]):
        return Features(reason="mercado_sem_oscilacao")
    if (bars[-1].epoch // 3600) % 24 in TOXIC_HOURS_UTC:
        return Features(reason="hora_toxica_utc")
    di14, _ = calc_wilder_di_adx(bars, 14)
    di20, adx20 = calc_wilder_di_adx(bars, 20)
    di24, _ = calc_wilder_di_adx(bars, 24)
    di28, _ = calc_wilder_di_adx(bars, 28)
    return Features(di14, di20, di24, di28, adx20, _resolve_h1_trend(bars, h1_bars), _candle_dir(bars[-1]))


def evaluate_features(features: Features, candidate: Candidate) -> tuple[str, str]:
    """Avalia gatilhos e preserva limiares estritos e veto de conflito."""
    if features.reason:
        return "SEM SINAL", features.reason
    scale = candidate.di_scale
    branches = (
        _branch_h1_di(features.h1_trend, features.di24, 40.0 * scale),
        _branch_h1_di(features.h1_trend, features.di14, 45.0 * scale),
        _branch_di_pullback(features.di14, features.direction, 45.0 * scale),
        _branch_di_continuation(features.di28, features.direction, 38.0 * scale),
        _branch_di_adx(features.di20, features.adx20, candidate.adx_min, 40.0 * scale),
    )
    hits = {branches[branch - 1] for branch in candidate.branches} - {None}
    if not hits:
        return "SEM SINAL", "sem_setup_persistent_trend"
    if len(hits) > 1:
        return "SEM SINAL", "conflito_persistent_trend"
    return hits.pop(), "setup_confirmado"


def evaluate_candidate(bars: list[Candle], h1_bars: list[Candle] | None, candidate: Candidate) -> tuple[str, str]:
    """Avalia uma hipotese usando os indicadores escalares de producao."""
    return evaluate_features(extract_features(bars, h1_bars), candidate)

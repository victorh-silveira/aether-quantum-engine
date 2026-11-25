"""Equivalencia causal e grade predeclarada dos candidatos de frequencia."""

from dataclasses import FrozenInstanceError, replace

import pytest
from hypothesis import (
    given,
    strategies as st,
)

from src.indicator.domain import Candle
from src.indicator.frequency_candidate import (
    Candidate,
    Features,
    evaluate_candidate,
    evaluate_features,
    extract_features,
    generate_candidates,
)
from src.indicator.strategy_persistent_trend import evaluate_persistent_trend_v3


def _bars(count=100, step=1.0):
    """Constroi velas fechadas validas com pressao direcional controlada."""
    return [
        Candle(index * 900, 1000 + index * step, 1002 + index * step, 998 + index * step, 1001 + index * step)
        for index in range(count)
    ]


def test_grid_is_fixed_deduplicated_and_immutable():
    candidates = generate_candidates()
    assert len(candidates) == 145
    assert len({candidate.version for candidate in candidates}) == 145
    assert len({candidate.label for candidate in candidates}) == 145
    assert Candidate() in candidates
    assert Candidate().version == Candidate(di_scale=1, adx_min=35).version
    with pytest.raises(FrozenInstanceError):
        candidates[0].di_scale = 0.25


@pytest.mark.parametrize("branches", [(), (6,), (2, 1), (1, 1), [1]])
def test_candidate_rejects_invalid_composition(branches):
    with pytest.raises(ValueError, match="composicao"):
        Candidate(branches=branches)


@pytest.mark.parametrize("scale,adx", [(0, 20), (-1, 20), (float("nan"), 20), (1, -1), (1, float("inf"))])
def test_candidate_rejects_invalid_thresholds(scale, adx):
    with pytest.raises(ValueError, match="limiar"):
        Candidate(di_scale=scale, adx_min=adx)


@pytest.mark.parametrize(
    "branch,field,threshold", [(1, "di24", 40), (2, "di14", 45), (3, "di14", 45), (4, "di28", 38), (5, "di20", 40)]
)
@pytest.mark.parametrize("sign", [-1, 1])
def test_strict_di_boundary_and_direction(branch, field, threshold, sign):
    features = Features(h1_trend=sign, direction=-sign if branch == 3 else sign, adx20=35)
    boundary = replace(features, **{field: sign * threshold})
    candidate = Candidate(branches=(branch,))
    assert evaluate_features(boundary, candidate)[0] == "SEM SINAL"
    above = replace(features, **{field: sign * (threshold + 1e-6)})
    assert evaluate_features(above, candidate) == ("CALL" if sign > 0 else "PUT", "setup_confirmado")


def test_adx_boundary_conflict_and_guard_reasons():
    assert evaluate_features(Features(di20=41, adx20=34.999), Candidate(branches=(5,)))[0] == "SEM SINAL"
    assert evaluate_features(Features(di20=41, adx20=35), Candidate(branches=(5,)))[0] == "CALL"
    conflicting = Features(di24=41, h1_trend=1, di28=-39, direction=-1)
    assert evaluate_features(conflicting, Candidate()) == ("SEM SINAL", "conflito_persistent_trend")
    assert evaluate_features(Features(reason="mercado_sem_oscilacao"), Candidate()) == (
        "SEM SINAL",
        "mercado_sem_oscilacao",
    )
    assert evaluate_features(Features(di20=float("nan")), Candidate())[0] == "SEM SINAL"


@given(st.integers(min_value=51, max_value=550), st.floats(min_value=-2, max_value=2, allow_nan=False))
def test_default_candidate_is_identical_to_production(count, step):
    bars = _bars(count, step)
    assert evaluate_candidate(bars, None, Candidate()) == evaluate_persistent_trend_v3(bars)


def test_extract_features_short_flat_and_toxic():
    assert extract_features([]).reason == "mercado_sem_oscilacao"
    bars = _bars()
    bars[-1] = Candle(bars[-1].epoch, 1000, 1000, 1000, 1000)
    assert extract_features(bars).reason == "mercado_sem_oscilacao"
    toxic = _bars(61)
    assert toxic[-1].epoch // 3600 % 24 == 15
    toxic = [replace(bar, epoch=bar.epoch + 12 * 3600) for bar in toxic]
    assert extract_features(toxic).reason == "hora_toxica_utc"


def test_h1_future_candles_cannot_change_candidate():
    bars = _bars()
    cutoff = bars[-1].epoch + 900
    h1 = [
        Candle(index * 3600 - 60 * 3600, 1000 + index, 1002 + index, 998 + index, 1001 + index) for index in range(60)
    ]
    future = Candle((cutoff // 3600) * 3600, 1e9, 1e9 + 2, 0, 1)
    assert evaluate_candidate(bars, h1, Candidate()) == evaluate_candidate(bars, [*h1, future], Candidate())
    assert evaluate_candidate(bars, h1, Candidate()) == evaluate_persistent_trend_v3(bars, h1)

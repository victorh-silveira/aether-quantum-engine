"""Testes da decisao organica de direcao (fluxo vs exaustao/fade)."""

from types import SimpleNamespace
from unittest.mock import patch

from src.domain.analytics.organic_direction import (
    _calc_adx,
    _calc_atr,
    _calc_bollinger_pct_b,
    _calc_cci,
    _calc_macd,
    _calc_rsi,
    _calc_stochastic,
    resolve_organic_direction,
)


def test_organic_direction_fallback():
    side, reason = resolve_organic_direction([], "CALL")
    assert side == "CALL"
    assert reason == "fallback_poucas_velas"


def test_internal_indicator_helpers():
    assert _calc_atr([]) == 1.0
    assert _calc_rsi([]) == 50.0
    assert 0.0 < _calc_rsi([10.0 + (i % 3) for i in range(25)]) < 100.0
    assert _calc_bollinger_pct_b([]) == 0.5
    assert _calc_stochastic([]) == (50.0, 50.0)
    assert _calc_macd([]) == (0.0, 0.0, 0.0)
    assert _calc_cci([]) == 0.0
    assert _calc_adx([]) == 20.0
    flat_c = [SimpleNamespace(open=10.0, high=10.0, low=10.0, close=10.0) for _ in range(20)]
    assert _calc_adx(flat_c) == 0.0


def test_organic_direction_call_trend_flow():
    candles = [SimpleNamespace(open=100.0 + i, high=101.5 + i, low=99.8 + i, close=101.0 + i) for i in range(50)]
    side, reason = resolve_organic_direction(candles, "CALL")
    assert side in ("CALL", "PUT")
    assert isinstance(reason, str)


def test_organic_direction_call_top_rejection():
    candles = [SimpleNamespace(open=100.0 + i, high=101.5 + i, low=99.8 + i, close=101.0 + i) for i in range(49)]
    candles.append(SimpleNamespace(open=150.0, high=165.0, low=149.0, close=151.0))
    side, reason = resolve_organic_direction(candles, "CALL")
    assert side == "PUT"
    assert reason == "fade_rejeicao_topo"


def test_organic_direction_call_exhaustion_and_extension():
    candles = [SimpleNamespace(open=100.0 + i, high=101.0 + i, low=99.0 + i, close=100.5 + i) for i in range(50)]
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=70.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.90),
        patch("src.domain.analytics.organic_direction._calc_stochastic", return_value=(40.0, 50.0)),
        patch("src.domain.analytics.organic_direction._calc_macd", return_value=(1.0, 0.5, 0.5)),
    ):
        side, reason = resolve_organic_direction(candles, "CALL")
        assert side == "PUT"
        assert reason == "fade_exaustao_sobrecompra"

    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=75.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.96),
        patch("src.domain.analytics.organic_direction._calc_stochastic", return_value=(60.0, 50.0)),
        patch("src.domain.analytics.organic_direction._calc_macd", return_value=(1.0, 0.5, 0.5)),
    ):
        side, reason = resolve_organic_direction(candles, "CALL")
        assert side == "PUT"
        assert reason == "fade_extensao_extrema"


def test_organic_direction_put_exhaustion_and_extension():
    candles = [SimpleNamespace(open=100.0 - i, high=101.0 - i, low=99.0 - i, close=99.5 - i) for i in range(50)]
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=30.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.10),
        patch("src.domain.analytics.organic_direction._calc_stochastic", return_value=(50.0, 40.0)),
        patch("src.domain.analytics.organic_direction._calc_macd", return_value=(-1.0, -0.5, -0.5)),
    ):
        side, reason = resolve_organic_direction(candles, "PUT")
        assert side == "CALL"
        assert reason == "bounce_exaustao_sobrevenda"

    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=25.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.04),
        patch("src.domain.analytics.organic_direction._calc_stochastic", return_value=(30.0, 40.0)),
        patch("src.domain.analytics.organic_direction._calc_macd", return_value=(-1.0, -0.5, -0.5)),
    ):
        side, reason = resolve_organic_direction(candles, "PUT")
        assert side == "CALL"
        assert reason == "bounce_extensao_extrema"


def test_organic_direction_put_healthy_flow():
    candles = [SimpleNamespace(open=100.0 - i, high=100.5 - i, low=98.8 - i, close=99.0 - i) for i in range(50)]
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=45.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.40),
        patch("src.domain.analytics.organic_direction._calc_stochastic", return_value=(30.0, 40.0)),
        patch("src.domain.analytics.organic_direction._calc_macd", return_value=(-1.0, -0.5, -0.5)),
        patch("src.domain.analytics.organic_direction._calc_cci", return_value=-50.0),
    ):
        side, reason = resolve_organic_direction(candles, "PUT")
        assert side == "PUT"
        assert reason == "fluxo_queda_saudavel"


def test_organic_direction_unknown_side():
    candles = [SimpleNamespace(open=100.0 + i, high=101.5 + i, low=99.8 + i, close=101.0 + i) for i in range(50)]
    side, reason = resolve_organic_direction(candles, "UNKNOWN_SIDE")
    assert side == "UNKNOWN_SIDE"
    assert reason == "direcao_preservada"


def test_organic_direction_flat_market_skip():
    candles = [SimpleNamespace(open=100.0, high=100.05, low=99.95, close=100.0) for _ in range(50)]
    side, reason = resolve_organic_direction(candles, "CALL")
    assert side == "SKIP"
    assert reason == "mercado_lateral_sem_volatilidade"


def test_organic_direction_bearish_momentum_reversal():
    candles = [SimpleNamespace(open=100.0 + i, high=101.5 + i, low=99.8 + i, close=101.0 + i) for i in range(48)]
    candles.append(SimpleNamespace(open=148.0, high=148.5, low=145.0, close=145.5))
    candles.append(SimpleNamespace(open=145.5, high=146.0, low=143.0, close=143.5))
    side, reason = resolve_organic_direction(candles, "CALL")
    assert side == "PUT"
    assert reason in ("fade_rejeicao_topo", "momentum_bearish_recuo", "fade_exaustao_sobrecompra")


def test_organic_direction_put_bottom_rejection():
    candles = [SimpleNamespace(open=100.0 - i, high=100.5 - i, low=98.8 - i, close=99.0 - i) for i in range(49)]
    candles.append(SimpleNamespace(open=50.0, high=51.0, low=35.0, close=49.0))
    side, reason = resolve_organic_direction(candles, "PUT")
    assert side == "CALL"
    assert reason == "bounce_rejeicao_fundo"


def test_organic_direction_bullish_momentum_reversal():
    candles = [SimpleNamespace(open=100.0 - i, high=100.5 - i, low=98.8 - i, close=99.0 - i) for i in range(48)]
    candles.append(SimpleNamespace(open=52.0, high=55.0, low=51.5, close=54.5))
    candles.append(SimpleNamespace(open=54.5, high=57.0, low=54.0, close=56.5))
    side, reason = resolve_organic_direction(candles, "PUT")
    assert side == "CALL"
    assert reason in ("bounce_rejeicao_fundo", "momentum_bullish_repique", "bounce_exaustao_sobrevenda")

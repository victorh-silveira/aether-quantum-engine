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
    _calc_session_vwap_bands,
    _calc_stochastic,
    resolve_organic_direction,
)


def _base_candles(n: int = 50, trend: float = 0.0) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            open=100.0 + i * trend,
            high=101.0 + i * trend,
            low=99.5 + i * trend,
            close=100.8 + i * trend,
        )
        for i in range(n)
    ]


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
    assert _calc_session_vwap_bands([]) == (0.0, 0.0, 0.0)


def test_organic_direction_flat_market_skip():
    candles = [SimpleNamespace(open=100.0, high=100.05, low=99.95, close=100.0) for _ in range(50)]
    side, reason = resolve_organic_direction(candles, "CALL")
    assert side == "SKIP"
    assert reason == "mercado_lateral_sem_volatilidade"


def test_organic_direction_call_exhaustion_veto():
    candles = _base_candles(50)
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=75.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.95),
    ):
        side, reason = resolve_organic_direction(candles, "CALL")
        assert side == "SKIP"
        assert reason == "veto_exaustao_sobrecompra"


def test_organic_direction_call_top_rejection_veto():
    candles = _base_candles(49)
    candles.append(SimpleNamespace(open=100.0, high=108.0, low=99.5, close=102.0))
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=50.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.50),
        patch("src.domain.analytics.organic_direction._calc_session_vwap_bands", return_value=(100.0, 110.0, 90.0)),
        patch("src.domain.analytics.organic_direction._calc_atr", return_value=4.0),
    ):
        side, reason = resolve_organic_direction(candles, "CALL")
        assert side == "SKIP"
        assert reason == "veto_rejeicao_topo"


def test_organic_direction_call_doji_veto():
    candles = _base_candles(49)
    candles.append(SimpleNamespace(open=100.0, high=100.6, low=99.6, close=100.05))
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=50.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.50),
        patch("src.domain.analytics.organic_direction._calc_session_vwap_bands", return_value=(100.0, 110.0, 90.0)),
        patch("src.domain.analytics.organic_direction._calc_atr", return_value=4.0),
    ):
        side, reason = resolve_organic_direction(candles, "CALL")
        assert side == "SKIP"
        assert reason == "veto_indecisao_doji"


def test_organic_direction_call_heavy_selling_pressure_veto():
    candles = _base_candles(49)
    candles.append(SimpleNamespace(open=100.0, high=100.2, low=97.0, close=97.2))
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=45.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.15),
        patch("src.domain.analytics.organic_direction._calc_session_vwap_bands", return_value=(100.0, 110.0, 90.0)),
        patch("src.domain.analytics.organic_direction._calc_macd", return_value=(0.0, 0.5, -0.5)),
        patch("src.domain.analytics.organic_direction._calc_cci", return_value=-150.0),
        patch("src.domain.analytics.organic_direction._calc_atr", return_value=4.0),
    ):
        side, reason = resolve_organic_direction(candles, "CALL")
        assert side == "SKIP"
        assert reason == "veto_pressao_vendedora_excessiva"


def test_organic_direction_call_pullback_support():
    candles = _base_candles(49)
    candles.append(SimpleNamespace(open=100.0, high=101.5, low=95.0, close=101.2))
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=50.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.50),
        patch("src.domain.analytics.organic_direction._calc_session_vwap_bands", return_value=(100.0, 110.0, 90.0)),
        patch("src.domain.analytics.organic_direction._calc_atr", return_value=4.0),
    ):
        side, reason = resolve_organic_direction(candles, "CALL")
        assert side == "CALL"
        assert reason == "pullback_suporte_retomada"


def test_organic_direction_call_healthy_flow():
    candles = _base_candles(49)
    candles.append(SimpleNamespace(open=100.0, high=102.0, low=99.8, close=101.8))
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=50.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.50),
        patch("src.domain.analytics.organic_direction._calc_session_vwap_bands", return_value=(100.0, 110.0, 90.0)),
        patch("src.domain.analytics.organic_direction._calc_stochastic", return_value=(40.0, 50.0)),
        patch("src.domain.analytics.organic_direction._calc_atr", return_value=4.0),
    ):
        side, reason = resolve_organic_direction(candles, "CALL")
        assert side == "CALL"
        assert reason == "fluxo_tendencia_saudavel"


def test_organic_direction_call_adverse_slope_veto():
    candles = _base_candles(50)
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=50.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.50),
        patch("src.domain.analytics.organic_direction._calc_session_vwap_bands", return_value=(100.0, 110.0, 90.0)),
        patch("src.domain.analytics.organic_direction._calc_ema", return_value=[100.0, 100.0, 95.0]),
        patch("src.domain.analytics.organic_direction._calc_atr", return_value=1.0),
    ):
        side, reason = resolve_organic_direction(candles, "CALL")
        assert side == "SKIP"
        assert reason == "veto_inclinacao_desfavoravel"


def test_organic_direction_put_exhaustion_veto():
    candles = _base_candles(50)
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=25.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.05),
    ):
        side, reason = resolve_organic_direction(candles, "PUT")
        assert side == "SKIP"
        assert reason == "veto_exaustao_sobrevenda"


def test_organic_direction_put_bottom_rejection_veto():
    candles = _base_candles(49)
    candles.append(SimpleNamespace(open=100.0, high=100.5, low=92.0, close=98.0))
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=50.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.50),
        patch("src.domain.analytics.organic_direction._calc_session_vwap_bands", return_value=(100.0, 110.0, 90.0)),
        patch("src.domain.analytics.organic_direction._calc_atr", return_value=4.0),
    ):
        side, reason = resolve_organic_direction(candles, "PUT")
        assert side == "SKIP"
        assert reason == "veto_rejeicao_fundo"


def test_organic_direction_put_doji_veto():
    candles = _base_candles(49)
    candles.append(SimpleNamespace(open=100.0, high=100.4, low=99.4, close=99.98))
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=50.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.50),
        patch("src.domain.analytics.organic_direction._calc_session_vwap_bands", return_value=(100.0, 110.0, 90.0)),
        patch("src.domain.analytics.organic_direction._calc_atr", return_value=4.0),
    ):
        side, reason = resolve_organic_direction(candles, "PUT")
        assert side == "SKIP"
        assert reason == "veto_indecisao_doji"


def test_organic_direction_put_heavy_buying_pressure_veto():
    candles = _base_candles(49)
    candles.append(SimpleNamespace(open=100.0, high=103.0, low=99.8, close=102.8))
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=55.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.85),
        patch("src.domain.analytics.organic_direction._calc_session_vwap_bands", return_value=(100.0, 110.0, 90.0)),
        patch("src.domain.analytics.organic_direction._calc_macd", return_value=(0.0, -0.5, 0.5)),
        patch("src.domain.analytics.organic_direction._calc_cci", return_value=150.0),
        patch("src.domain.analytics.organic_direction._calc_atr", return_value=4.0),
    ):
        side, reason = resolve_organic_direction(candles, "PUT")
        assert side == "SKIP"
        assert reason == "veto_pressao_compradora_excessiva"


def test_organic_direction_put_pullback_resistance():
    candles = _base_candles(49)
    candles.append(SimpleNamespace(open=100.0, high=107.0, low=98.5, close=98.8))
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=50.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.50),
        patch("src.domain.analytics.organic_direction._calc_session_vwap_bands", return_value=(100.0, 110.0, 90.0)),
        patch("src.domain.analytics.organic_direction._calc_atr", return_value=4.0),
    ):
        side, reason = resolve_organic_direction(candles, "PUT")
        assert side == "PUT"
        assert reason == "pullback_resistencia_retomada"


def test_organic_direction_put_healthy_flow():
    candles = _base_candles(49)
    candles.append(SimpleNamespace(open=100.0, high=100.2, low=98.0, close=98.2))
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=45.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.40),
        patch("src.domain.analytics.organic_direction._calc_session_vwap_bands", return_value=(100.0, 110.0, 90.0)),
        patch("src.domain.analytics.organic_direction._calc_stochastic", return_value=(60.0, 50.0)),
        patch("src.domain.analytics.organic_direction._calc_atr", return_value=4.0),
    ):
        side, reason = resolve_organic_direction(candles, "PUT")
        assert side == "PUT"
        assert reason == "fluxo_queda_saudavel"


def test_organic_direction_put_adverse_slope_veto():
    candles = _base_candles(50)
    with (
        patch("src.domain.analytics.organic_direction._calc_rsi", return_value=50.0),
        patch("src.domain.analytics.organic_direction._calc_bollinger_pct_b", return_value=0.50),
        patch("src.domain.analytics.organic_direction._calc_session_vwap_bands", return_value=(100.0, 110.0, 90.0)),
        patch("src.domain.analytics.organic_direction._calc_ema", return_value=[100.0, 100.0, 105.0]),
        patch("src.domain.analytics.organic_direction._calc_atr", return_value=1.0),
    ):
        side, reason = resolve_organic_direction(candles, "PUT")
        assert side == "SKIP"
        assert reason == "veto_inclinacao_desfavoravel"


def test_organic_direction_unknown_side():
    candles = _base_candles(50)
    side, reason = resolve_organic_direction(candles, "UNKNOWN_SIDE")
    assert side == "UNKNOWN_SIDE"
    assert reason == "direcao_preservada"

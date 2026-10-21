"""Testes unitarios para o logger de indicadores de ciclo."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.orchestrator.cycle_indicators_logger import (
    _calc_cci,
    _calc_cmo,
    _calc_ema,
    _calc_hurst_exponent,
    _calc_keltner_pct_b,
    _calc_macd,
    _calc_roc,
    _calc_stochastic,
    _calc_williams_r,
    log_cycle_indicators,
)
from src.domain.models.market_data import Candle


def test_calc_ema_single_and_multi():
    """Valida calculo da EMA."""
    assert _calc_ema([10.0], 20) == [10.0]
    res = _calc_ema([10.0, 11.0, 12.0], 2)
    assert len(res) == 3
    assert res[-1] > 10.0


def test_calc_macd_short_and_normal():
    """Valida MACD com historico curto e completo."""
    assert _calc_macd([10.0] * 5, 12, 26, 9) == (0.0, 0.0, 0.0)
    series = [10.0 + i * 0.1 for i in range(40)]
    m_val, s_val, hist = _calc_macd(series, 12, 26, 9)
    assert isinstance(m_val, float)
    assert isinstance(s_val, float)
    assert isinstance(hist, float)


def test_calc_stochastic_short_and_flat():
    """Valida Estocastico com historico insuficiente e barras estagnadas."""
    b_short = [Candle("R_25", 10.0, 10.0, 10.0, 10.0, datetime.fromtimestamp(i * 300), i * 300) for i in range(5)]
    assert _calc_stochastic(b_short, 14, 3) == (50.0, 50.0)
    b_flat = [Candle("R_25", 10.0, 10.0, 10.0, 10.0, datetime.fromtimestamp(i * 300), i * 300) for i in range(20)]
    k, d = _calc_stochastic(b_flat, 14, 3)
    assert k == 50.0
    assert d == 50.0


def test_calc_additional_indicators_coverage():
    """Valida CCI, CMO, Williams %R, ROC, Keltner e Hurst com dados vazios e normais."""
    b_short = [Candle("R_25", 10.0, 10.0, 10.0, 10.0, datetime.fromtimestamp(i * 300), i * 300) for i in range(5)]
    b_trend = [
        Candle(
            "R_25",
            10.0 + i,
            11.0 + i,
            9.0 + i,
            10.5 + i,
            datetime.fromtimestamp(i * 300),
            i * 300,
        )
        for i in range(25)
    ]
    assert _calc_cci([], 14) == 0.0
    assert _calc_cci(b_short, 14) == 0.0
    assert isinstance(_calc_cci(b_trend, 14), float)

    assert _calc_cmo([], 14) == 0.0
    assert _calc_cmo([10.0] * 10, 14) == 0.0
    assert isinstance(_calc_cmo([10.0 + i for i in range(20)], 14), float)

    assert _calc_williams_r([], 14) == -50.0
    assert _calc_williams_r(b_short, 14) == -50.0
    assert isinstance(_calc_williams_r(b_trend, 14), float)

    assert _calc_roc([], 10) == 0.0
    assert _calc_roc([0.0] * 15, 10) == 0.0
    assert isinstance(_calc_roc([10.0 + i for i in range(15)], 10), float)

    assert _calc_keltner_pct_b([], 20) == 0.5
    assert _calc_keltner_pct_b(b_short, 20) == 0.5
    assert isinstance(_calc_keltner_pct_b(b_trend, 20), float)

    assert _calc_hurst_exponent([], 32) == 0.5
    assert _calc_hurst_exponent([10.0] * 40, 32) == 0.5
    assert 0.0 <= _calc_hurst_exponent([10.0 + i * 0.1 for i in range(40)], 32) <= 1.0


def test_log_cycle_indicators_early_returns():
    """Valida retornos antecipados sem config, sem stream ou sem candles suficientes."""
    orch = SimpleNamespace(config={}, logger=MagicMock())
    log_cycle_indicators(orch)
    assert orch.logger.info.call_count == 0

    orch.config = {"pullback_strategy": {"enabled": True}}
    orch.stream = None
    orch.stream_handler = None
    log_cycle_indicators(orch)
    assert orch.logger.info.call_count == 0

    orch.stream = SimpleNamespace(micro_candles={"R_25": []})
    log_cycle_indicators(orch)
    assert orch.logger.info.call_count == 0


def test_log_cycle_indicators_alta_and_baixa_with_rejections():
    """Valida emissao de log com quebra de linha para regimes de ALTA e BAIXA com rejeicoes."""
    orch = SimpleNamespace(
        config={"pullback_strategy": {"enabled": True}, "anchor": "R_25"},
        logger=MagicMock(),
        _active_cycle_id="0001",
        symbols=["R_25"],
    )
    candles_up = [
        Candle(
            "R_25",
            100.0 + i * 0.5,
            101.0 + i * 0.5,
            99.0 + i * 0.5,
            100.5 + i * 0.5,
            datetime.fromtimestamp(i * 300),
            i * 300,
        )
        for i in range(60)
    ]
    candles_up[-1] = Candle("R_25", 130.0, 135.0, 129.5, 130.2, datetime.fromtimestamp(59 * 300), 59 * 300)
    orch.stream = SimpleNamespace(micro_candles={"R_25": candles_up})
    log_cycle_indicators(orch)
    assert orch.logger.info.call_count == 1
    msg = orch.logger.info.call_args[0][0]
    assert "ALTA" in msg
    assert "\n" in msg
    assert "PRECO" in msg
    assert "OSCILADORES" in msg

    candles_down = [
        Candle(
            "R_25",
            200.0 - i * 0.5,
            201.0 - i * 0.5,
            199.0 - i * 0.5,
            199.5 - i * 0.5,
            datetime.fromtimestamp(i * 300),
            i * 300,
        )
        for i in range(60)
    ]
    candles_down[-1] = Candle("R_25", 170.0, 170.5, 165.0, 169.8, datetime.fromtimestamp(59 * 300), 59 * 300)
    orch.stream.micro_candles["R_25"] = candles_down
    log_cycle_indicators(orch)
    assert orch.logger.info.call_count == 2
    msg_down = orch.logger.info.call_args[0][0]
    assert "BAIXA" in msg_down


def test_log_cycle_indicators_flat_bars_zero_variances():
    """Valida resiliencia a barras com variancia zero e tr_s zero."""
    orch = SimpleNamespace(
        config={"pullback_strategy": {"enabled": True}, "anchor": "R_25"},
        logger=MagicMock(),
        _active_cycle_id="0002",
        symbols=["R_25"],
    )
    flat_bars = [
        Candle("R_25", 100.0, 100.0, 100.0, 100.0, datetime.fromtimestamp(i * 300), i * 300) for i in range(60)
    ]
    orch.stream = SimpleNamespace(micro_candles={"R_25": flat_bars})
    log_cycle_indicators(orch)
    assert orch.logger.info.call_count == 1


def test_log_cycle_indicators_veto_ansi_highlighting():
    """Valida presenca de codigos ANSI e liberacao sem vetos (organico)."""
    orch = SimpleNamespace(
        config={"pullback_strategy": {"enabled": True}, "anchor": "R_25"},
        logger=MagicMock(),
        _active_cycle_id="0003",
        symbols=["R_25"],
    )
    candles = [
        Candle(
            "R_25",
            100.0,
            101.0,
            99.0,
            100.0,
            datetime.fromtimestamp(i * 300),
            i * 300,
        )
        for i in range(60)
    ]
    orch.stream = SimpleNamespace(micro_candles={"R_25": candles})
    log_cycle_indicators(orch)
    assert orch.logger.info.call_count == 1
    msg = orch.logger.info.call_args[0][0]
    assert "\033[92m" in msg or "\033[0m" in msg
    assert "LIBERADO SEM VETOS (ORGANICO)" in msg

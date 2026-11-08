"""Contratos do setup fixo por velas fechadas."""

from unittest.mock import patch

from indicator.domain import Candle
from indicator.pullback import MIN_BARS, VERSION, evaluate, version_for_variant


def call_bars(period=300):
    """Constroi tendencia causal com recuo e confirmacao validos."""
    bars = [
        Candle(
            index * period,
            100 + index * 0.1,
            100 + index * 0.1 + 0.08,
            100 + index * 0.1 - 0.08,
            100 + index * 0.1 + 0.04,
        )
        for index in range(MIN_BARS - 2)
    ]
    base = 100 + (MIN_BARS - 2) * 0.1 - 0.25
    bars.append(Candle((MIN_BARS - 2) * period, base, base + 0.1, base - 0.75, base + 0.05))
    bars.append(Candle((MIN_BARS - 1) * period, base + 0.05, base + 0.25, base + 0.04, base + 0.2))
    return bars


def test_call_put_and_period_isolation():
    for period in (300, 900, 3600):
        bars = call_bars(period)
        assert len(bars) == MIN_BARS
        result = evaluate(bars, period, MIN_BARS * period)
        assert (result.side, result.reason, result.version) == ("CALL", "setup_confirmado", VERSION)
        mirrored = [Candle(bar.epoch, 200 - bar.open, 200 - bar.low, 200 - bar.high, 200 - bar.close) for bar in bars]
        assert evaluate(mirrored, period, MIN_BARS * period).side == "PUT"


def test_sem_sinal_on_missing_forming_stale_or_incomplete():
    bars = call_bars()
    assert evaluate(bars, 60, MIN_BARS * 300).reason == "periodo_invalido"
    assert evaluate(bars[:-1], 300, MIN_BARS * 300).reason == "historico_insuficiente"
    assert evaluate(bars, 900, MIN_BARS * 300).reason == "vela_invalida"
    gapped = list(bars)
    gapped[20] = Candle(21 * 300, *tuple(vars(gapped[20])[name] for name in ("open", "high", "low", "close")))
    assert evaluate(gapped, 300, MIN_BARS * 300).reason == "lacuna"
    assert evaluate(bars, 300, (MIN_BARS - 1) * 300 + 10).reason == "vela_em_formacao"
    assert evaluate(bars, 300, (MIN_BARS + 1) * 300).reason == "dados_atrasados"
    shifted = list(bars)
    old = shifted[-2]
    shifted[-2] = Candle(old.epoch, old.open, old.high, old.open - 0.15, old.close)
    assert evaluate(shifted, 300, MIN_BARS * 300).reason == "padrao_incompleto"
    broken = list(bars)
    last = broken[-1]
    broken[-1] = Candle(last.epoch, last.open, last.high, last.low, last.open + 0.001)
    assert evaluate(broken, 300, MIN_BARS * 300).reason == "corpo_fora_faixa"
    flat = [Candle(bar.epoch, 100, 100, 100, 100) for bar in bars]
    assert evaluate(flat, 300, MIN_BARS * 300).reason == "atr_invalido"


def test_predeclared_variants_are_causal_and_not_operational_default():
    bars = call_bars()
    assert evaluate(bars, 300, MIN_BARS * 300).side == "CALL"
    assert evaluate(bars, 300, MIN_BARS * 300, "ema50_slope").side == "CALL"
    assert evaluate(bars, 300, MIN_BARS * 300, "countertrend_retrace").side == "SEM SINAL"
    assert evaluate(bars, 300, MIN_BARS * 300, "entry_distance").side in ("CALL", "SEM SINAL")
    assert evaluate(bars, 300, MIN_BARS * 300, "unknown").reason == "variante_invalida"
    mirrored = [Candle(bar.epoch, 200 - bar.open, 200 - bar.low, 200 - bar.high, 200 - bar.close) for bar in bars]
    assert evaluate(mirrored, 300, MIN_BARS * 300, "ema50_slope").side == "PUT"
    assert evaluate(mirrored, 300, MIN_BARS * 300, "countertrend_retrace").side == "SEM SINAL"
    assert evaluate(bars, 300, MIN_BARS * 300, "pool_rsi_bb").side in ("CALL", "PUT", "SEM SINAL")
    assert version_for_variant("momentum_burst").startswith("momentum-1-")
    assert evaluate(bars, 300, MIN_BARS * 300, "momentum_burst").side in ("CALL", "PUT", "SEM SINAL")
    with (
        patch("indicator.pullback._rsi", return_value=60.0),
        patch("indicator.pullback._adx", return_value=(30.0, 35.0, 5.0)),
        patch("indicator.pullback._stochastic", return_value=(60.0, 50.0)),
        patch("indicator.pullback._macd", return_value=(1.0, 0.5, 0.5)),
        patch("indicator.pullback._cci", return_value=50.0),
        patch("indicator.pullback._cmo", return_value=10.0),
        patch("indicator.pullback._williams_r", return_value=-20.0),
        patch("indicator.pullback._roc", return_value=0.5),
        patch("indicator.pullback._bollinger_pct_b", return_value=0.6),
        patch("indicator.pullback._keltner_pct_b", return_value=0.6),
        patch("indicator.pullback._hurst_exponent", return_value=0.6),
    ):
        assert evaluate(bars, 300, MIN_BARS * 300, "momentum_burst").side == "CALL"

    with (
        patch("indicator.pullback._rsi", return_value=40.0),
        patch("indicator.pullback._adx", return_value=(30.0, 5.0, 35.0)),
        patch("indicator.pullback._stochastic", return_value=(40.0, 50.0)),
        patch("indicator.pullback._macd", return_value=(-1.0, -0.5, -0.5)),
        patch("indicator.pullback._cci", return_value=-50.0),
        patch("indicator.pullback._cmo", return_value=-10.0),
        patch("indicator.pullback._williams_r", return_value=-80.0),
        patch("indicator.pullback._roc", return_value=-0.5),
        patch("indicator.pullback._bollinger_pct_b", return_value=0.4),
        patch("indicator.pullback._keltner_pct_b", return_value=0.4),
        patch("indicator.pullback._hurst_exponent", return_value=0.6),
    ):
        assert evaluate(mirrored, 300, MIN_BARS * 300, "momentum_burst").side == "PUT"


def test_rsi_and_bollinger_helpers():
    from indicator.pullback import _adx, _bollinger_pct_b, _macd, _rsi, _stochastic

    assert _rsi([]) == 50.0
    assert _rsi([10.0] * 20) == 50.0
    assert _rsi(list(range(20))) == 100.0
    assert _bollinger_pct_b([]) == 0.5
    assert _bollinger_pct_b([10.0] * 5) == 0.5
    assert _bollinger_pct_b([10.0] * 20) == 0.5
    assert 0.0 <= _bollinger_pct_b(list(range(25))) <= 1.0
    assert _adx([]) == (0.0, 0.0, 0.0)
    flat_bars = [Candle(i * 300, 100.0, 100.0, 100.0, 100.0) for i in range(20)]
    assert _adx(flat_bars) == (0.0, 0.0, 0.0)
    trend_bars = [Candle(i * 300, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i) for i in range(25)]
    dx, pdi, mdi = _adx(trend_bars)
    assert pdi > mdi
    assert dx >= 0.0
    assert _macd([10.0] * 5) == (0.0, 0.0, 0.0)
    m_val, s_val, hist = _macd([10.0 + i * 0.1 for i in range(40)])
    assert isinstance(m_val, float)
    assert isinstance(s_val, float)
    assert isinstance(hist, float)
    assert _stochastic([], 14, 3) == (50.0, 50.0)
    stoch_flat = [Candle(i * 300, 100.0, 100.0, 100.0, 100.0) for i in range(20)]
    k, d = _stochastic(stoch_flat, 14, 3)
    assert k == 50.0
    assert d == 50.0
    stoch_trend = [Candle(i * 300, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i) for i in range(20)]
    kt, dt = _stochastic(stoch_trend, 14, 3)
    assert isinstance(kt, float)
    assert isinstance(dt, float)

    from indicator.pullback import _cci, _cmo, _hurst_exponent, _keltner_pct_b, _roc, _williams_r

    assert _cci([], 14) == 0.0
    assert _cci(flat_bars, 14) == 0.0
    assert isinstance(_cci(trend_bars, 14), float)

    assert _cmo([], 14) == 0.0
    assert _cmo([10.0] * 20, 14) == 0.0
    assert isinstance(_cmo(list(range(20)), 14), float)

    assert _williams_r([], 14) == -50.0
    assert _williams_r(flat_bars, 14) == -50.0
    assert isinstance(_williams_r(trend_bars, 14), float)

    assert _roc([], 10) == 0.0
    assert _roc([0.0] * 15, 10) == 0.0
    assert isinstance(_roc([10.0 + i for i in range(15)], 10), float)

    assert _keltner_pct_b([], 20) == 0.5
    assert _keltner_pct_b(flat_bars, 20) == 0.5
    assert isinstance(_keltner_pct_b(trend_bars, 20), float)

    assert _hurst_exponent([], 32) == 0.5
    assert _hurst_exponent([10.0] * 40, 32) == 0.5
    assert 0.0 <= _hurst_exponent([10.0 + i * 0.1 for i in range(40)], 32) <= 1.0

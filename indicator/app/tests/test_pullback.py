"""Contratos do setup fixo por velas fechadas."""

from indicator.domain import Candle
from indicator.pullback import MIN_BARS, VERSION, evaluate


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
    bars.append(Candle((MIN_BARS - 1) * period, base + 0.05, base + 0.23, base + 0.04, base + 0.2))
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

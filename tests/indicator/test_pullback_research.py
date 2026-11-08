"""Pesquisa cronologica sem selecionar parametros pelo mesmo teste."""

from indicator.domain import Candle
from indicator.pullback import MIN_BARS, SetupResult
from indicator.pullback_research import VARIANTS, compare_variants


def test_comparison_requires_history_and_keeps_holdout(monkeypatch):
    assert compare_variants([], 300) == {}
    bars = [Candle(i * 300, 100, 101, 99, 100 + (i % 3 - 1) * 0.1) for i in range(5 * MIN_BARS)]
    seen = []

    def setup(window, period, now, variant):
        seen.append((now, variant))
        return SetupResult("CALL" if variant == "base" else "SEM SINAL", "teste")

    monkeypatch.setattr("indicator.pullback_research.evaluate", setup)
    report = compare_variants(bars, 300)
    assert set(report) == set(VARIANTS)
    assert len(report["base"]["janelas"]) == 4
    assert report["base"]["teste_final"] == report["base"]["janelas"][-1]
    assert report["base"]["qualifica_contrato"] is False
    assert report["base"]["teste_final"]["setups"] > 0
    assert report["ema50_slope"]["teste_final"]["setups"] == 0
    assert all(now % 300 == 0 for now, _ in seen)
